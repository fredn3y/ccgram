"""Codex history sync — create lazy Telegram topics for desktop Codex chats."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
from telegram import Bot
from telegram.error import TelegramError

from ..config import config
from ..thread_router import thread_router
from ..telegram_sender import split_message
from ..utils import atomic_write_json, task_done_callback
from .message_sender import safe_send
from .resume_command import ResumeEntry, _create_resume_window, scan_all_sessions
from .resume_topics_command import _bound_session_ids, _topic_name_for_resume

logger = structlog.get_logger()

_state_lock = asyncio.Lock()


@dataclass(frozen=True, slots=True)
class PendingCodexTopic:
    """A Telegram topic that represents a Codex session not yet running in tmux."""

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


@dataclass
class CodexHistoryState:
    """Persisted sync state for Codex Desktop history topics."""

    initialized: bool
    seen_session_ids: set[str]
    pending_topics: dict[str, PendingCodexTopic]


def _topic_key(user_id: int, thread_id: int) -> str:
    return f"{user_id}:{thread_id}"


def _state_path() -> Path:
    return config.codex_history_sync_file


def _load_state() -> CodexHistoryState:
    path = _state_path()
    if not path.exists():
        return CodexHistoryState(False, set(), {})

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        logger.warning("Failed to read Codex history sync state; starting fresh")
        return CodexHistoryState(False, set(), {})

    pending: dict[str, PendingCodexTopic] = {}
    for key, item in data.get("pending_topics", {}).items():
        if not isinstance(item, dict):
            continue
        try:
            pending[str(key)] = PendingCodexTopic(
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
            )
        except (KeyError, TypeError, ValueError):
            continue

    return CodexHistoryState(
        initialized=bool(data.get("initialized", False)),
        seen_session_ids={str(sid) for sid in data.get("seen_session_ids", [])},
        pending_topics=pending,
    )


def _save_state(state: CodexHistoryState) -> None:
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


async def codex_history_sync_loop(bot: Bot) -> None:
    """Poll Codex history and create Telegram topics for new desktop sessions."""
    if not config.codex_history_sync_enabled:
        return

    logger.info(
        "Codex history sync started (interval %.1fs)",
        config.codex_history_sync_interval,
    )
    while True:
        try:
            await sync_codex_history_once(bot)
        except Exception:
            logger.exception("Codex history sync iteration failed")
        await asyncio.sleep(config.codex_history_sync_interval)


def start_codex_history_sync(bot: Bot) -> asyncio.Task[None] | None:
    """Start the Codex history sync loop when configured for the Codex provider."""
    if not config.codex_history_sync_enabled:
        return None
    task = asyncio.create_task(codex_history_sync_loop(bot))
    task.add_done_callback(task_done_callback)
    return task


async def sync_codex_history_once(bot: Bot) -> None:
    """Create lazy Telegram topics for newly discovered Codex sessions."""
    target = _default_target()
    if target is None:
        return

    sessions = scan_all_sessions("codex")

    async with _state_lock:
        state = _load_state()
        if state.pending_topics:
            await _sync_pending_history(bot, state)

        if not sessions:
            _save_state(state)
            return

        session_ids = {entry.session_id for entry in sessions}
        if not state.initialized:
            state.initialized = True
            state.seen_session_ids.update(session_ids)
            _save_state(state)
            logger.info(
                "Codex history sync seeded %d existing session(s)", len(session_ids)
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
    state: CodexHistoryState,
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
        if not entry.cwd or not Path(entry.cwd).is_dir():
            state.seen_session_ids.add(entry.session_id)
            continue
        candidates.append(entry)
    return candidates


def _has_user_summary(entry: ResumeEntry) -> bool:
    """Return True once Codex history has a usable user prompt summary."""
    summary = (entry.summary or "").strip()
    return bool(summary and summary != entry.session_id[:12])


@dataclass(frozen=True, slots=True)
class _SyncTarget:
    user_id: int
    chat_id: int


def _default_target() -> _SyncTarget | None:
    if not config.allowed_users:
        return None

    user_id = sorted(config.allowed_users)[0]
    chat_id = config.group_id
    if chat_id is None:
        for candidate in thread_router.group_chat_ids.values():
            if candidate < 0:
                chat_id = candidate
                break
    if chat_id is None:
        logger.debug("Codex history sync has no Telegram group target yet")
        return None
    return _SyncTarget(user_id=user_id, chat_id=chat_id)


async def _create_pending_topic(
    bot: Bot,
    target: _SyncTarget,
    entry: ResumeEntry,
) -> tuple[str, PendingCodexTopic] | None:
    topic_name = _topic_name_for_resume(entry)
    try:
        topic = await bot.create_forum_topic(
            chat_id=target.chat_id,
            name=topic_name,
        )
    except TelegramError as exc:
        logger.warning(
            "Failed to auto-create Codex history topic for session %s: %s",
            entry.session_id,
            exc,
        )
        return None

    pending = PendingCodexTopic(
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
        "Auto-created lazy Codex topic '%s' (thread=%d) for session %s",
        topic_name,
        topic.message_thread_id,
        entry.session_id,
    )
    return key, pending


async def _sync_pending_history(bot: Bot, state: CodexHistoryState) -> None:
    """Send newly available Codex transcript messages into pending topics."""
    for key, pending in list(state.pending_topics.items()):
        state.pending_topics[key] = await _send_pending_history(bot, pending)


async def _send_pending_history(
    bot: Bot,
    pending: PendingCodexTopic,
) -> PendingCodexTopic:
    messages, new_offset = _read_codex_event_history(
        pending.transcript_path,
        pending.history_offset,
    )
    if not messages or new_offset == pending.history_offset:
        return pending

    for message in messages:
        for chunk in split_message(message, max_length=3900):
            sent = await safe_send(
                bot,
                pending.chat_id,
                chunk,
                message_thread_id=pending.thread_id,
                disable_notification=True,
            )
            if sent is None:
                return pending

    return PendingCodexTopic(
        session_id=pending.session_id,
        summary=pending.summary,
        cwd=pending.cwd,
        transcript_path=pending.transcript_path,
        user_id=pending.user_id,
        chat_id=pending.chat_id,
        thread_id=pending.thread_id,
        topic_name=pending.topic_name,
        created_at=pending.created_at,
        history_offset=new_offset,
    )


def _read_codex_event_history(
    transcript_path: str,
    start_offset: int,
) -> tuple[list[str], int]:
    """Read user/assistant event messages from a Codex transcript."""
    messages: list[str] = []
    offset = max(0, start_offset)
    try:
        with open(transcript_path, "rb") as file:
            file.seek(offset)
            for raw_line in file:
                offset += len(raw_line)
                message = _parse_codex_event_history_line(raw_line)
                if message:
                    messages.append(message)
    except OSError:
        return [], start_offset
    return messages, offset


def _parse_codex_event_history_line(raw_line: bytes) -> str:
    try:
        entry = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""

    if not isinstance(entry, dict) or entry.get("type") != "event_msg":
        return ""
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return ""

    event_type = payload.get("type")
    if event_type == "user_message":
        text = str(payload.get("message") or "").strip()
        return f"\U0001f464 {text}" if text else ""
    if event_type == "agent_message":
        text = str(payload.get("message") or "").strip()
        return text
    return ""


async def activate_pending_codex_topic(
    user_id: int,
    thread_id: int,
    chat_id: int,
) -> str | None:
    """Resume a pending Codex history topic into tmux and bind it.

    Returns the created window_id when activation succeeds, otherwise None.
    """
    async with _state_lock:
        state = _load_state()
        pending = state.pending_topics.get(_topic_key(user_id, thread_id))
        if not pending:
            return None

    success, message, created_wname, created_wid, _provider_name = (
        await _create_resume_window(
            user_id,
            thread_id,
            pending.session_id,
            pending.cwd,
            pending.transcript_path,
            preferred_window_name=pending.topic_name,
        )
    )
    if not success:
        logger.warning(
            "Failed to activate pending Codex topic %d for session %s: %s",
            thread_id,
            pending.session_id,
            message,
        )
        return None

    thread_router.bind_thread(
        user_id,
        thread_id,
        created_wid,
        window_name=created_wname,
    )
    thread_router.set_group_chat_id(user_id, thread_id, chat_id)

    async with _state_lock:
        state = _load_state()
        state.pending_topics.pop(_topic_key(user_id, thread_id), None)
        state.seen_session_ids.add(pending.session_id)
        _save_state(state)

    logger.info(
        "Activated pending Codex topic %d as window %s for session %s",
        thread_id,
        created_wid,
        pending.session_id,
    )
    return created_wid


async def forget_pending_codex_topic(user_id: int, thread_id: int) -> None:
    """Drop pending metadata when a lazy topic is closed before activation."""
    async with _state_lock:
        state = _load_state()
        if state.pending_topics.pop(_topic_key(user_id, thread_id), None) is None:
            return
        _save_state(state)
