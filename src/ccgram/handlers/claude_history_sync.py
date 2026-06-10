"""Claude Code history sync — create lazy Telegram topics for Claude Code chats.

Mirrors the Codex history sync: new Claude Code sessions found under
``~/.claude/projects`` get a lazy Telegram forum topic, transcript history is
backfilled incrementally by byte offset, and replies in the topic continue the
conversation with a headless ``claude -p --resume <session-id>`` turn (the
Claude CLI keeps the same session id and appends to the same transcript file,
so the watcher picks up the response like any other desktop-side output).

Unlike Codex there is no app-server: the headless resume turn IS the submit
path, and topic titles come from ``ai-title`` lines that Claude Code writes
into the transcript itself.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import structlog
from telegram import Bot
from telegram.error import TelegramError

from ..config import config
from ..providers import resolve_launch_command
from ..telegram_sender import split_rendered_message
from ..thread_router import thread_router
from ..utils import atomic_write_json, task_done_callback
from .codex_history_sync import (
    _consume_submitted_prompt_echo,
    _default_target,
    _SyncTarget,
)
from .message_sender import safe_send
from .resume_command import ResumeEntry, scan_all_sessions
from .resume_topics_command import _bound_session_ids, _topic_name_for_resume
from .topic_emoji import sync_topic_name, update_stored_topic_name

logger = structlog.get_logger()

_state_lock = asyncio.Lock()
_TELEGRAM_TOPIC_NAME_MAX = 128
_HISTORY_CHUNKS_PER_SYNC = 20

# In-flight headless resume turns keyed by topic key — one turn per topic.
_active_turns: dict[str, asyncio.Task[None]] = {}


@dataclass(frozen=True, slots=True)
class PendingClaudeTopic:
    """A Telegram topic that mirrors a Claude Code session."""

    session_id: str
    summary: str
    cwd: str
    transcript_path: str
    user_id: int
    chat_id: int
    thread_id: int
    topic_name: str
    created_at: float
    history_offset: int = 0
    submitted_prompt_echoes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingClaudeAction:
    """Result of handling a message in a pending Claude topic."""

    status: str
    message: str = ""


@dataclass
class ClaudeHistoryState:
    """Persisted sync state for Claude Code history topics."""

    initialized: bool
    seen_session_ids: set[str]
    pending_topics: dict[str, PendingClaudeTopic]


@dataclass(frozen=True, slots=True)
class ClaudeHistoryEntry:
    """A parsed user/assistant message or title from a Claude transcript."""

    role: str  # "user", "agent", or "title"
    text: str
    formatted: str
    end_offset: int = 0


def _topic_key(user_id: int, thread_id: int) -> str:
    return f"{user_id}:{thread_id}"


def _state_path() -> Path:
    return config.claude_history_sync_file


def _load_state() -> ClaudeHistoryState:
    path = _state_path()
    if not path.exists():
        return ClaudeHistoryState(False, set(), {})

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        logger.warning("Failed to read Claude history sync state; starting fresh")
        return ClaudeHistoryState(False, set(), {})

    pending: dict[str, PendingClaudeTopic] = {}
    for key, item in data.get("pending_topics", {}).items():
        if not isinstance(item, dict):
            continue
        try:
            pending[str(key)] = PendingClaudeTopic(
                session_id=str(item["session_id"]),
                summary=str(item.get("summary", "")),
                cwd=str(item["cwd"]),
                transcript_path=str(item.get("transcript_path", "")),
                user_id=int(item["user_id"]),
                chat_id=int(item["chat_id"]),
                thread_id=int(item["thread_id"]),
                topic_name=str(item.get("topic_name", "")),
                created_at=float(item.get("created_at", 0.0)),
                history_offset=int(item.get("history_offset", 0)),
                submitted_prompt_echoes=tuple(
                    str(prompt)
                    for prompt in item.get("submitted_prompt_echoes", [])
                    if str(prompt).strip()
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue

    return ClaudeHistoryState(
        initialized=bool(data.get("initialized", False)),
        seen_session_ids={str(sid) for sid in data.get("seen_session_ids", [])},
        pending_topics=pending,
    )


def _save_state(state: ClaudeHistoryState) -> None:
    atomic_write_json(
        _state_path(),
        {
            "initialized": state.initialized,
            "seen_session_ids": sorted(state.seen_session_ids),
            "pending_topics": {
                key: asdict(topic) for key, topic in state.pending_topics.items()
            },
        },
    )


async def claude_history_sync_loop(bot: Bot) -> None:
    """Poll Claude Code history and create Telegram topics for new sessions."""
    if not config.claude_history_sync_enabled:
        return

    logger.info(
        "Claude history sync started (interval %.1fs)",
        config.claude_history_sync_interval,
    )
    while True:
        try:
            await sync_claude_history_once(bot)
        except Exception:
            logger.exception("Claude history sync iteration failed")
        await asyncio.sleep(config.claude_history_sync_interval)


def start_claude_history_sync(bot: Bot) -> asyncio.Task[None] | None:
    """Start the Claude history sync loop when enabled."""
    if not config.claude_history_sync_enabled:
        return None
    task = asyncio.create_task(claude_history_sync_loop(bot))
    task.add_done_callback(task_done_callback)
    return task


async def sync_claude_history_once(bot: Bot) -> None:
    """Create lazy Telegram topics for newly discovered Claude sessions."""
    target = _default_target()
    if target is None:
        return

    sessions = scan_all_sessions("claude")

    async with _state_lock:
        state = _load_state()
        if state.pending_topics:
            await _sync_pending_history(bot, state)
            _save_state(state)

        if not sessions:
            return

        session_ids = {entry.session_id for entry in sessions}
        if not state.initialized:
            state.initialized = True
            state.seen_session_ids.update(session_ids)
            _save_state(state)
            logger.info(
                "Claude history sync seeded %d existing session(s)", len(session_ids)
            )
            return

        candidates = _new_session_candidates(sessions, state, target.user_id)
        if not candidates:
            _save_state(state)
            return

        for entry in candidates:
            created = await _create_pending_topic(bot, target, entry)
            if created:
                key, topic = created
                topic = await _send_pending_history(bot, topic)
                state.pending_topics[key] = topic
                state.seen_session_ids.add(entry.session_id)

        _save_state(state)


def _new_session_candidates(
    sessions: list[ResumeEntry],
    state: ClaudeHistoryState,
    user_id: int,
) -> list[ResumeEntry]:
    pending_session_ids = {
        topic.session_id for topic in state.pending_topics.values()
    }
    bound_session_ids = _bound_session_ids(user_id)
    state.seen_session_ids.update(bound_session_ids)

    candidates: list[ResumeEntry] = []
    for entry in reversed(sessions):
        if entry.session_id in state.seen_session_ids:
            continue
        if entry.session_id in pending_session_ids:
            continue
        if entry.session_id in bound_session_ids:
            continue
        if not _has_user_summary(entry):
            continue
        if not entry.transcript_path:
            continue
        if not entry.cwd or not Path(entry.cwd).is_dir():
            state.seen_session_ids.add(entry.session_id)
            continue
        candidates.append(entry)
    return candidates


def _has_user_summary(entry: ResumeEntry) -> bool:
    """Return True once Claude history has a usable user prompt summary."""
    summary = (entry.summary or "").strip()
    return bool(summary and summary != entry.session_id[:12])


async def _create_pending_topic(
    bot: Bot,
    target: _SyncTarget,
    entry: ResumeEntry,
) -> tuple[str, PendingClaudeTopic] | None:
    topic_name = _topic_name_for_resume(entry)
    try:
        topic = await bot.create_forum_topic(
            chat_id=target.chat_id,
            name=topic_name,
        )
    except TelegramError as exc:
        logger.warning(
            "Failed to auto-create Claude history topic for session %s: %s",
            entry.session_id,
            exc,
        )
        return None

    pending = PendingClaudeTopic(
        session_id=entry.session_id,
        summary=entry.summary,
        cwd=entry.cwd,
        transcript_path=entry.transcript_path,
        user_id=target.user_id,
        chat_id=target.chat_id,
        thread_id=topic.message_thread_id,
        topic_name=topic_name,
        created_at=time.time(),
    )
    key = _topic_key(target.user_id, topic.message_thread_id)
    logger.info(
        "Auto-created lazy Claude topic '%s' (thread=%d) for session %s",
        topic_name,
        topic.message_thread_id,
        entry.session_id,
    )
    return key, pending


async def _sync_pending_history(bot: Bot, state: ClaudeHistoryState) -> None:
    """Send newly available Claude transcript messages into pending topics."""
    for key, pending in list(state.pending_topics.items()):
        state.pending_topics[key] = await _send_pending_history(bot, pending)
        _save_state(state)


async def _send_pending_history(
    bot: Bot,
    pending: PendingClaudeTopic,
) -> PendingClaudeTopic:
    entries, new_offset = _read_claude_history_entries(
        pending.transcript_path,
        pending.history_offset,
    )
    if not entries or new_offset == pending.history_offset:
        return pending

    submitted_prompt_echoes = pending.submitted_prompt_echoes
    sent_chunks = 0
    for entry in entries:
        if entry.role == "title":
            pending = await _apply_title_entry(bot, pending, entry)
            continue

        if entry.role == "user":
            matched, submitted_prompt_echoes = _consume_submitted_prompt_echo(
                submitted_prompt_echoes,
                entry.text,
            )
            if matched:
                pending = replace(
                    pending,
                    history_offset=entry.end_offset or pending.history_offset,
                    submitted_prompt_echoes=submitted_prompt_echoes,
                )
                continue

        for chunk in split_rendered_message(entry.formatted):
            sent = await safe_send(
                bot,
                pending.chat_id,
                chunk,
                message_thread_id=pending.thread_id,
                disable_notification=True,
            )
            if sent is None:
                return pending
            sent_chunks += 1

        pending = replace(
            pending,
            history_offset=entry.end_offset or pending.history_offset,
            submitted_prompt_echoes=submitted_prompt_echoes,
        )
        if sent_chunks >= _HISTORY_CHUNKS_PER_SYNC:
            return pending

    return replace(
        pending,
        history_offset=new_offset,
        submitted_prompt_echoes=submitted_prompt_echoes,
    )


async def _apply_title_entry(
    bot: Bot,
    pending: PendingClaudeTopic,
    entry: ClaudeHistoryEntry,
) -> PendingClaudeTopic:
    """Rename the Telegram topic from a transcript ``ai-title`` line."""
    new_name = _normalize_topic_name(entry.text)
    advanced = replace(
        pending,
        history_offset=entry.end_offset or pending.history_offset,
    )
    if not new_name or new_name == pending.topic_name:
        return advanced

    await sync_topic_name(bot, pending.chat_id, pending.thread_id, new_name)
    update_stored_topic_name(pending.chat_id, pending.thread_id, new_name)
    logger.info(
        "Synced Claude session title to Telegram topic %d: %r",
        pending.thread_id,
        new_name,
    )
    return replace(advanced, topic_name=new_name)


def _normalize_topic_name(name: str) -> str:
    clean = " ".join(str(name or "").split())
    return clean[:_TELEGRAM_TOPIC_NAME_MAX].rstrip(" -") if clean else ""


# --- transcript parsing -------------------------------------------------------


def _read_claude_history_entries(
    transcript_path: str,
    start_offset: int,
) -> tuple[list[ClaudeHistoryEntry], int]:
    """Read parsed user/assistant/title entries from a Claude transcript."""
    entries: list[ClaudeHistoryEntry] = []
    offset = max(0, start_offset)
    try:
        with open(transcript_path, "rb") as file:
            file.seek(offset)
            for raw_line in file:
                offset += len(raw_line)
                entry = _parse_claude_history_entry(raw_line)
                if entry is not None:
                    entries.append(replace(entry, end_offset=offset))
    except OSError:
        return [], start_offset
    return entries, offset


_SKIPPED_USER_PREFIXES = (
    "<command-",
    "<local-command",
    "<system-reminder",
)


def _parse_claude_history_entry(raw_line: bytes) -> ClaudeHistoryEntry | None:
    try:
        entry = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(entry, dict):
        return None

    entry_type = entry.get("type")
    if entry_type == "ai-title":
        return _title_history_entry(entry)
    if entry_type not in ("user", "assistant") or _is_hidden_entry(entry):
        return None

    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    if entry_type == "user":
        return _user_history_entry(message)
    return _agent_history_entry(message)


def _is_hidden_entry(entry: dict) -> bool:
    return bool(
        entry.get("isMeta")
        or entry.get("isSidechain")
        or entry.get("isCompactSummary")
        or entry.get("isVisibleInTranscriptOnly")
    )


def _title_history_entry(entry: dict) -> ClaudeHistoryEntry | None:
    title = str(entry.get("aiTitle") or "").strip()
    if not title:
        return None
    return ClaudeHistoryEntry(role="title", text=title, formatted=title)


def _user_history_entry(message: dict) -> ClaudeHistoryEntry | None:
    text = _extract_user_text(message.get("content"))
    if not text:
        return None
    return ClaudeHistoryEntry(role="user", text=text, formatted=f"\U0001f464 {text}")


def _agent_history_entry(message: dict) -> ClaudeHistoryEntry | None:
    text = _extract_assistant_text(message.get("content"))
    if not text:
        return None
    return ClaudeHistoryEntry(role="agent", text=text, formatted=text)


def _extract_user_text(content: object) -> str:
    """Extract displayable text from a Claude user message content payload."""
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = [
            str(block.get("text") or "").strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "\n".join(part for part in parts if part).strip()
    else:
        return ""

    if not text:
        return ""
    if text.startswith(_SKIPPED_USER_PREFIXES):
        return ""
    return text


def _extract_assistant_text(content: object) -> str:
    """Extract displayable text from a Claude assistant content payload."""
    if not isinstance(content, list):
        return ""
    parts = [
        str(block.get("text") or "").strip()
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n\n".join(part for part in parts if part).strip()


# --- reply path ----------------------------------------------------------------


async def submit_to_pending_claude_topic(
    user_id: int,
    thread_id: int,
    chat_id: int,
    text: str,
    bot: Bot,
) -> PendingClaudeAction:
    """Continue a mirrored Claude session with a headless resume turn."""
    key = _topic_key(user_id, thread_id)
    async with _state_lock:
        state = _load_state()
        pending = state.pending_topics.get(key)
        if not pending:
            return PendingClaudeAction("not_pending")

    active = _active_turns.get(key)
    if active and not active.done():
        return PendingClaudeAction("busy")

    if not Path(pending.cwd).is_dir():
        return PendingClaudeAction(
            "failed",
            message=f"Session working directory is gone: {pending.cwd}",
        )

    thread_router.set_group_chat_id(user_id, thread_id, chat_id)
    await _record_submitted_prompt(key, pending, text)

    task = asyncio.create_task(_run_claude_turn(pending, text, bot))
    task.add_done_callback(task_done_callback)
    task.add_done_callback(lambda _t: _active_turns.pop(key, None))
    _active_turns[key] = task
    return PendingClaudeAction("submitted")


async def _record_submitted_prompt(
    key: str,
    pending: PendingClaudeTopic,
    text: str,
) -> None:
    """Persist the prompt echo before the turn writes it into the transcript."""
    async with _state_lock:
        state = _load_state()
        current = state.pending_topics.get(key, pending)
        state.pending_topics[key] = replace(
            current,
            submitted_prompt_echoes=(*current.submitted_prompt_echoes, text),
        )
        state.seen_session_ids.add(current.session_id)
        _save_state(state)


def _claude_turn_command(session_id: str, text: str) -> list[str]:
    base = shlex.split(resolve_launch_command("claude"))
    cmd = [
        *base,
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--resume",
        session_id,
    ]
    if config.claude_history_model:
        cmd += ["--model", config.claude_history_model]
    cmd.append(text)
    return cmd


async def _run_claude_turn(
    pending: PendingClaudeTopic,
    text: str,
    bot: Bot,
) -> None:
    """Run one headless Claude resume turn; the watcher mirrors its output."""
    cmd = _claude_turn_command(pending.session_id, text)
    logger.info(
        "Starting Claude resume turn for topic %d (session %s)",
        pending.thread_id,
        pending.session_id,
    )
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=pending.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        await _report_turn_error(bot, pending, f"Could not start Claude: {exc}")
        return

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=config.claude_history_turn_timeout,
        )
    except asyncio.TimeoutError:
        process.kill()
        await _report_turn_error(
            bot,
            pending,
            f"Claude turn timed out after {config.claude_history_turn_timeout:.0f}s.",
        )
        return

    if process.returncode != 0:
        detail = _turn_error_detail(stdout, stderr)
        await _report_turn_error(
            bot,
            pending,
            f"Claude turn failed (exit {process.returncode}). {detail}".strip(),
        )
        return

    result = _parse_turn_result(stdout)
    if result.get("is_error"):
        detail = str(result.get("result") or "engine error")
        await _report_turn_error(bot, pending, detail[:500])
        return

    logger.info(
        "Claude resume turn finished for topic %d (session %s)",
        pending.thread_id,
        pending.session_id,
    )


def _parse_turn_result(stdout: bytes) -> dict:
    try:
        parsed = json.loads(stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _turn_error_detail(stdout: bytes, stderr: bytes) -> str:
    result = _parse_turn_result(stdout)
    if result.get("result"):
        return str(result["result"])[:500]
    tail = stderr.decode("utf-8", errors="replace").strip()
    return tail[-500:] if tail else ""


async def _report_turn_error(
    bot: Bot,
    pending: PendingClaudeTopic,
    detail: str,
) -> None:
    logger.warning(
        "Claude resume turn error for topic %d: %s",
        pending.thread_id,
        detail,
    )
    await safe_send(
        bot,
        pending.chat_id,
        f"⚠️ {detail}" if detail else "⚠️ Claude turn failed.",
        message_thread_id=pending.thread_id,
    )


async def forget_pending_claude_topic(user_id: int, thread_id: int) -> None:
    """Drop pending metadata when a lazy topic is closed."""
    async with _state_lock:
        state = _load_state()
        if state.pending_topics.pop(_topic_key(user_id, thread_id), None) is None:
            return
        _save_state(state)
