"""Resume command — browse and resume past agent sessions.

Implements /resume: scans provider-specific session data, groups sessions by
project directory, and shows a paginated inline keyboard. On selection, creates
a tmux window with the provider's native resume command and binds the current
topic.

Key functions:
  - resume_command: /resume handler
  - handle_resume_command_callback: callback dispatcher for resume UI
  - scan_all_sessions: discover all resumable sessions across all projects
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import structlog
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..config import config
from ..providers import get_provider, get_provider_for_window, resolve_launch_command
from .. import window_query
from ..session import session_manager
from ..session_map import session_map_sync
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from ..window_state_store import CCGRAM_CREATED_WINDOW_ORIGIN
from ..utils import read_session_metadata_from_jsonl
from .callback_data import CB_RESUME_CANCEL, CB_RESUME_PAGE, CB_RESUME_PICK
from .callback_helpers import get_thread_id
from .callback_registry import register
from .message_sender import safe_edit, safe_reply
from .topic_emoji import format_topic_name_for_mode, get_stored_topic_name
from .user_state import RESUME_SESSIONS

logger = structlog.get_logger()

_SESSIONS_PER_PAGE = 6
_CODEX_SCAN_LINES = 80

_IndexParseError = (json.JSONDecodeError, OSError)


@dataclass
class ResumeEntry:
    """A resumable session discovered from project directories."""

    session_id: str
    summary: str
    cwd: str
    transcript_path: str = ""


def _stored_topic_name_for_query(query: CallbackQuery, thread_id: int) -> str | None:
    """Return the user-created Telegram topic title for a callback topic."""
    message = query.message
    chat = message.chat if message else None
    if not chat or chat.type not in ("group", "supergroup"):
        return None
    return get_stored_topic_name(chat.id, thread_id)


def scan_all_sessions(provider_name: str | None = None) -> list[ResumeEntry]:
    """Scan project directories for resumable sessions.

    Claude sessions live under ``~/.claude/projects``. Codex sessions live
    under ``~/.codex/sessions/YYYY/MM/DD``. Unsupported providers keep the
    legacy Claude-compatible scan path.

    Returns entries sorted by file mtime (most recent first),
    deduplicated by session_id.
    """
    provider = _resolve_resume_provider(provider_name)
    if provider == "codex":
        return _scan_codex_sessions()
    return _scan_claude_sessions()


def _resolve_resume_provider(provider_name: str | None) -> str:
    """Return the provider whose history should be scanned."""
    raw_provider = provider_name
    if raw_provider is None:
        raw_provider = getattr(config, "provider_name", "claude")
    return str(raw_provider or "claude").lower()


def _scan_claude_sessions() -> list[ResumeEntry]:
    """Scan Claude-style project directories for resumable sessions."""
    if not config.claude_projects_path.exists():
        return []

    candidates: list[tuple[float, ResumeEntry]] = []
    seen_ids: set[str] = set()

    for project_dir in config.claude_projects_path.iterdir():
        if not project_dir.is_dir():
            continue

        # Try legacy sessions-index.json first
        index_file = project_dir / "sessions-index.json"
        if index_file.exists():
            _scan_index_file(index_file, seen_ids, candidates)

        # Pick up bare JSONL files (no index required)
        _scan_bare_jsonl(project_dir, seen_ids, candidates)

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [entry for _, entry in candidates]


def _codex_sessions_dir() -> Path:
    """Resolve Codex's session transcript directory."""
    codex_home = os.getenv("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "sessions"
    return Path.home() / ".codex" / "sessions"


def _scan_codex_sessions() -> list[ResumeEntry]:
    """Scan Codex JSONL transcripts for resumable sessions."""
    sessions_dir = _codex_sessions_dir()
    if not sessions_dir.exists():
        return []

    candidates: list[tuple[float, ResumeEntry]] = []
    seen_ids: set[str] = set()

    try:
        jsonl_files = sessions_dir.rglob("*.jsonl")
    except OSError:
        return []

    for jsonl_file in jsonl_files:
        session = _read_codex_resume_entry(jsonl_file)
        if session is None or session.session_id in seen_ids:
            continue

        try:
            mtime = jsonl_file.stat().st_mtime
        except OSError:
            mtime = 0.0

        seen_ids.add(session.session_id)
        candidates.append((mtime, session))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [entry for _, entry in candidates]


def _read_codex_resume_entry(jsonl_file: Path) -> ResumeEntry | None:
    """Read enough of a Codex transcript to build a resume picker entry."""
    session_id = ""
    cwd = ""
    summary = ""
    fallback_summary = ""

    for data, payload in _iter_codex_payloads(jsonl_file):
        if data.get("type") == "session_meta":
            session_id = _first_str(payload.get("id"), session_id)
            cwd = _first_str(payload.get("cwd"), cwd)

        if not summary:
            summary = _extract_codex_event_user_summary(data, payload)

        if not fallback_summary:
            fallback_summary = _extract_codex_item_user_summary(data, payload)

        if session_id and cwd and summary:
            break

    if not session_id or not cwd:
        return None
    return ResumeEntry(
        session_id,
        summary or fallback_summary or session_id[:12],
        cwd,
        str(jsonl_file),
    )


def _iter_codex_payloads(jsonl_file: Path):
    """Yield parsed Codex transcript entries with dict payloads."""
    try:
        with jsonl_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= _CODEX_SCAN_LINES:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue

                payload = data.get("payload")
                if not isinstance(payload, dict):
                    continue

                yield data, payload
    except OSError:
        return


def _first_str(value: object, fallback: str = "") -> str:
    """Return value if it is a non-empty string, otherwise fallback."""
    return value if isinstance(value, str) and value else fallback


def _extract_codex_event_user_summary(data: dict, payload: dict) -> str:
    """Extract a user-facing prompt from a Codex event entry."""
    if data.get("type") != "event_msg" or payload.get("type") != "user_message":
        return ""

    text = _first_str(payload.get("message"))
    if text:
        return text[:80]
    return _extract_codex_text(payload.get("text_elements"))


def _extract_codex_item_user_summary(data: dict, payload: dict) -> str:
    """Extract fallback user text from lower-level Codex transcript items."""
    if (
        data.get("type") == "response_item"
        and payload.get("type") == "message"
        and payload.get("role") == "user"
    ):
        return _extract_codex_text(payload.get("content"))

    if data.get("type") == "input_item":
        return _extract_codex_text(payload.get("content"))

    return ""


def _extract_codex_text(content: object) -> str:
    """Extract plain text from Codex content blocks."""
    if isinstance(content, str):
        return content[:80]
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)

    return "".join(parts)[:80]


def _scan_index_file(
    index_file: Path,
    seen_ids: set[str],
    candidates: list[tuple[float, ResumeEntry]],
) -> None:
    """Scan a sessions-index.json for resumable sessions."""
    try:
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
    except _IndexParseError:
        return

    original_path = index_data.get("originalPath", "")
    for entry in index_data.get("entries", []):
        session_id = entry.get("sessionId", "")
        full_path = entry.get("fullPath", "")
        if not session_id or not full_path or session_id in seen_ids:
            continue

        file_path = Path(full_path)
        if not file_path.exists():
            continue

        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = 0.0

        cwd = entry.get("projectPath", original_path)
        summary = (
            entry.get("summary", "") or entry.get("firstPrompt", "") or session_id[:12]
        )
        seen_ids.add(session_id)
        candidates.append((mtime, ResumeEntry(session_id, summary, cwd)))


def _scan_bare_jsonl(
    project_dir: Path,
    seen_ids: set[str],
    candidates: list[tuple[float, ResumeEntry]],
) -> None:
    """Scan bare JSONL files not covered by a sessions-index."""
    try:
        jsonl_iter = project_dir.glob("*.jsonl")
    except OSError:
        return

    for jsonl_file in jsonl_iter:
        session_id = jsonl_file.stem
        if session_id in seen_ids:
            continue

        cwd, summary = read_session_metadata_from_jsonl(jsonl_file)
        if not cwd:
            continue

        try:
            mtime = jsonl_file.stat().st_mtime
        except OSError:
            mtime = 0.0

        seen_ids.add(session_id)
        candidates.append(
            (mtime, ResumeEntry(session_id, summary or session_id[:12], cwd))
        )


def _build_resume_keyboard(
    sessions: list[dict[str, str]],
    page: int = 0,
) -> InlineKeyboardMarkup:
    """Build inline keyboard for resume session picker with pagination."""
    total = len(sessions)
    start = page * _SESSIONS_PER_PAGE
    end = min(start + _SESSIONS_PER_PAGE, total)
    page_sessions = sessions[start:end]

    rows: list[list[InlineKeyboardButton]] = []
    current_cwd = ""
    for idx_offset, entry in enumerate(page_sessions):
        global_idx = start + idx_offset
        cwd = entry.get("cwd", "")
        # Show project header when cwd changes
        if cwd != current_cwd:
            current_cwd = cwd
            short_path = Path(cwd).name if cwd else "unknown"
            rows.append(
                [
                    InlineKeyboardButton(
                        f"\U0001f4c1 {short_path}",
                        callback_data="noop",
                    )
                ]
            )
        label = entry.get("summary", "")[:40] or entry["session_id"][:12]
        rows.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=f"{CB_RESUME_PICK}{global_idx}"[:64],
                )
            ]
        )

    # Pagination row
    nav_buttons: list[InlineKeyboardButton] = []
    if page > 0:
        nav_buttons.append(
            InlineKeyboardButton(
                "\u2b05 Prev",
                callback_data=f"{CB_RESUME_PAGE}{page - 1}"[:64],
            )
        )
    total_pages = (total + _SESSIONS_PER_PAGE - 1) // _SESSIONS_PER_PAGE
    if page < total_pages - 1:
        nav_buttons.append(
            InlineKeyboardButton(
                "Next \u27a1",
                callback_data=f"{CB_RESUME_PAGE}{page + 1}"[:64],
            )
        )
    nav_buttons.append(
        InlineKeyboardButton("\u2716 Cancel", callback_data=CB_RESUME_CANCEL)
    )
    rows.append(nav_buttons)

    return InlineKeyboardMarkup(rows)


async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /resume — show all resumable sessions grouped by project."""
    if not update.message:
        return

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await safe_reply(
            update.message,
            "\u274c Please use /resume in a named topic.",
        )
        return

    # Check resume capability using per-window provider (or global fallback)
    window_id = thread_router.get_window_for_thread(user.id, thread_id)
    provider = (
        get_provider_for_window(
            window_id,
            provider_name=window_query.get_window_provider(window_id),
        )
        if window_id
        else get_provider()
    )
    if not provider.capabilities.supports_resume:
        await safe_reply(
            update.message,
            "\u274c Resume is not supported by the current provider.",
        )
        return

    sessions = scan_all_sessions(provider.capabilities.name)
    if not sessions:
        await safe_reply(update.message, "\u274c No past sessions found.")
        return

    session_dicts = [
        {
            "session_id": s.session_id,
            "summary": s.summary,
            "cwd": s.cwd,
            "transcript_path": s.transcript_path,
        }
        for s in sessions
    ]
    if context.user_data is not None:
        context.user_data[RESUME_SESSIONS] = session_dicts

    keyboard = _build_resume_keyboard(session_dicts, page=0)
    await safe_reply(
        update.message,
        "\u23ea Select a session to resume:",
        reply_markup=keyboard,
    )


async def handle_resume_command_callback(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Dispatch resume command callbacks."""
    if data.startswith(CB_RESUME_PICK):
        await _handle_pick(query, user_id, data, update, context)
    elif data.startswith(CB_RESUME_PAGE):
        await _handle_page(query, user_id, data, update, context)
    elif data == CB_RESUME_CANCEL:
        await _handle_cancel(query, context)


async def _create_resume_window(
    user_id: int,
    thread_id: int,
    session_id: str,
    cwd: str,
    transcript_path: str = "",
    preferred_window_name: str | None = None,
) -> tuple[bool, str, str, str, str]:
    """Unbind old window, create a new one with resume args.

    Returns (success, message, window_name, window_id, provider_name).
    """
    old_window_id = thread_router.get_window_for_thread(user_id, thread_id)
    if old_window_id:
        thread_router.unbind_thread(user_id, thread_id)
        from .polling_strategies import lifecycle_strategy

        lifecycle_strategy.clear_dead_notification(user_id, thread_id)

    if old_window_id:
        old_view = session_manager.view_window(old_window_id)
        provider = get_provider_for_window(
            old_window_id, provider_name=old_view.provider_name if old_view else None
        )
        approval_mode = old_view.approval_mode if old_view else "normal"
    else:
        provider = get_provider()
        approval_mode = "normal"
    launch_args = provider.make_launch_args(resume_id=session_id)
    launch_command = resolve_launch_command(
        provider.capabilities.name, approval_mode=approval_mode
    )
    create_kwargs = {
        "agent_args": launch_args,
        "launch_command": launch_command,
    }
    if provider.capabilities.name == "codex" and preferred_window_name:
        create_kwargs["window_name"] = preferred_window_name
    success, message, created_wname, created_wid = await tmux_manager.create_window(
        cwd, **create_kwargs
    )
    if success:
        session_manager.set_window_origin(created_wid, CCGRAM_CREATED_WINDOW_ORIGIN)
        session_manager.set_window_provider(created_wid, provider.capabilities.name)
        session_manager.set_window_approval_mode(created_wid, approval_mode)
        if provider.capabilities.supports_hook:
            await session_map_sync.wait_for_session_map_entry(created_wid)
        elif transcript_path:
            session_map_sync.register_hookless_session(
                window_id=created_wid,
                session_id=session_id,
                cwd=cwd,
                transcript_path=transcript_path,
                provider_name=provider.capabilities.name,
            )
            await asyncio.to_thread(
                session_map_sync.write_hookless_session_map,
                window_id=created_wid,
                session_id=session_id,
                cwd=cwd,
                transcript_path=transcript_path,
                provider_name=provider.capabilities.name,
            )

    return success, message, created_wname, created_wid, provider.capabilities.name


async def _handle_pick(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle session selection from the resume picker."""
    idx_str = data[len(CB_RESUME_PICK) :]
    try:
        idx = int(idx_str)
    except ValueError:
        await query.answer("Invalid selection", show_alert=True)
        return

    thread_id = get_thread_id(update)
    if thread_id is None:
        await query.answer("Use in a topic", show_alert=True)
        return

    stored = context.user_data.get(RESUME_SESSIONS) if context.user_data else None
    if not stored or idx < 0 or idx >= len(stored):
        await query.answer("Invalid session index", show_alert=True)
        return

    picked = stored[idx]
    session_id = picked["session_id"]
    cwd = picked.get("cwd", "")
    transcript_path = picked.get("transcript_path", "")

    if not cwd or not Path(cwd).is_dir():
        await safe_edit(query, "\u274c Project directory no longer exists.")
        _clear_resume_state(context.user_data)
        await query.answer("Failed")
        return

    success, message, created_wname, created_wid, provider_name = (
        await _create_resume_window(
            user_id,
            thread_id,
            session_id,
            cwd,
            transcript_path,
            preferred_window_name=_stored_topic_name_for_query(query, thread_id),
        )
    )
    if not success:
        await safe_edit(query, f"\u274c {message}")
        _clear_resume_state(context.user_data)
        await query.answer("Failed")
        return

    thread_router.bind_thread(
        user_id, thread_id, created_wid, window_name=created_wname
    )

    # Store group chat_id for routing
    chat = query.message.chat if query.message else None
    if chat and chat.type in ("group", "supergroup"):
        thread_router.set_group_chat_id(user_id, thread_id, chat.id)

    # Rename topic to match the window for providers that benefit from lifecycle badges.
    if provider_name != "codex":
        try:
            await context.bot.edit_forum_topic(
                chat_id=thread_router.resolve_chat_id(user_id, thread_id),
                message_thread_id=thread_id,
                name=format_topic_name_for_mode(
                    created_wname, session_manager.get_approval_mode(created_wid)
                ),
            )
        except TelegramError as e:
            logger.debug("Failed to rename topic: %s", e)

    summary_short = picked.get("summary", "")[:40]
    await safe_edit(
        query,
        f"\u2705 Resuming session: {summary_short}\n\U0001f4c2 `{cwd}`",
    )
    _clear_resume_state(context.user_data)
    await query.answer("Resumed")


async def _handle_page(
    query: CallbackQuery,
    _user_id: int,
    data: str,
    _update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle pagination in resume picker."""
    page_str = data[len(CB_RESUME_PAGE) :]
    try:
        page = int(page_str)
    except ValueError:
        await query.answer("Invalid page", show_alert=True)
        return

    stored = context.user_data.get(RESUME_SESSIONS) if context.user_data else None
    if not stored:
        await query.answer("No sessions available", show_alert=True)
        return

    keyboard = _build_resume_keyboard(stored, page=page)
    await safe_edit(
        query,
        "\u23ea Select a session to resume:",
        reply_markup=keyboard,
    )
    await query.answer()


async def _handle_cancel(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle cancel in resume picker."""
    _clear_resume_state(context.user_data)
    await safe_edit(query, "Resume cancelled.")
    await query.answer("Cancelled")


def _clear_resume_state(user_data: dict | None) -> None:
    """Remove resume-related keys from user_data."""
    if user_data is None:
        return
    user_data.pop(RESUME_SESSIONS, None)


# --- Registry dispatch entry point ---


@register(CB_RESUME_PICK, CB_RESUME_PAGE, CB_RESUME_CANCEL)
async def _dispatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    assert query is not None and query.data is not None and user is not None
    await handle_resume_command_callback(query, user.id, query.data, update, context)
