"""Resume topics command — create forum topics for resumable sessions."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog
from telegram import Update
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from ..config import config
from ..providers import get_provider
from ..session import session_manager
from ..thread_router import thread_router
from ..tmux_manager import tmux_manager
from .message_sender import safe_reply
from .resume_command import ResumeEntry, _create_resume_window, scan_all_sessions

logger = structlog.get_logger()

_MAX_RESUME_TOPICS = 20
_TELEGRAM_TOPIC_NAME_MAX = 128


@dataclass(frozen=True, slots=True)
class ResumeTopicPlan:
    """Sessions that should become topics plus skip counts for the reply."""

    candidates: list[ResumeEntry]
    already_bound: int = 0
    missing_cwd: int = 0


@dataclass(frozen=True, slots=True)
class ResumeTopicsTarget:
    """Telegram target details needed to create and bind topics."""

    user_id: int
    chat_id: int


@dataclass(frozen=True, slots=True)
class ResumeTopicsResult:
    """Outcome counts after creating resume topics."""

    created: int = 0
    rebound: int = 0
    failed: int = 0
    limited: int = 0


async def resume_topics_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle /resume_topics — create one topic per dormant resumable session."""
    target, error = _resolve_target(update)
    if error and update.message:
        await safe_reply(update.message, error)
        return
    if not target or not update.message:
        return

    provider = get_provider()
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

    plan = _plan_resume_topics(sessions, _bound_session_ids(target.user_id))
    if not plan.candidates:
        await safe_reply(
            update.message,
            _format_resume_topics_reply(
                created=0,
                rebound=0,
                failed=0,
                already_bound=plan.already_bound,
                missing_cwd=plan.missing_cwd,
                limited=0,
            ),
            disable_notification=True,
        )
        return

    result = await _materialize_resume_topics(context, target, plan.candidates)
    await safe_reply(
        update.message,
        _format_resume_topics_reply(
            created=result.created,
            rebound=result.rebound,
            failed=result.failed,
            already_bound=plan.already_bound,
            missing_cwd=plan.missing_cwd,
            limited=result.limited,
        ),
        disable_notification=True,
    )


def _resolve_target(update: Update) -> tuple[ResumeTopicsTarget | None, str]:
    if not update.message:
        return None, ""

    user = update.effective_user
    if not user or not config.is_user_allowed(user.id):
        return None, ""

    chat = update.effective_chat or update.message.chat
    if not chat or chat.type not in ("group", "supergroup"):
        return (
            None,
            "\u274c Please use /resume_topics in your private Telegram forum group.",
        )

    if not getattr(chat, "is_forum", False):
        return (
            None,
            "\u274c This group needs forum topics enabled before I can create resume topics.",
        )

    return ResumeTopicsTarget(user_id=user.id, chat_id=chat.id), ""


def _plan_resume_topics(
    sessions: list[ResumeEntry],
    bound_session_ids: set[str],
) -> ResumeTopicPlan:
    """Filter resumable sessions down to dormant sessions with live directories."""
    candidates: list[ResumeEntry] = []
    already_bound = 0
    missing_cwd = 0

    for entry in sessions:
        if entry.session_id in bound_session_ids:
            already_bound += 1
            continue
        if not entry.cwd or not Path(entry.cwd).is_dir():
            missing_cwd += 1
            continue
        candidates.append(entry)

    return ResumeTopicPlan(candidates, already_bound, missing_cwd)


async def _materialize_resume_topics(
    context: ContextTypes.DEFAULT_TYPE,
    target: ResumeTopicsTarget,
    candidates: list[ResumeEntry],
) -> ResumeTopicsResult:
    live_unbound = await _live_unbound_windows_by_session(target.user_id)
    selected = candidates[:_MAX_RESUME_TOPICS]
    limited = max(0, len(candidates) - len(selected))
    created = 0
    rebound = 0
    failed = 0

    for entry in selected:
        outcome = await _materialize_resume_topic(
            context,
            target,
            entry,
            live_unbound.get(entry.session_id),
        )
        if outcome == "created":
            created += 1
        elif outcome == "rebound":
            rebound += 1
        else:
            failed += 1

    return ResumeTopicsResult(
        created=created,
        rebound=rebound,
        failed=failed,
        limited=limited,
    )


async def _materialize_resume_topic(
    context: ContextTypes.DEFAULT_TYPE,
    target: ResumeTopicsTarget,
    entry: ResumeEntry,
    live_window_id: str | None,
) -> Literal["created", "rebound", "failed"]:
    topic_name = _topic_name_for_resume(entry)
    try:
        topic = await context.bot.create_forum_topic(
            chat_id=target.chat_id,
            name=topic_name,
        )
    except TelegramError as exc:
        logger.warning(
            "Failed to create resume topic for session %s: %s",
            entry.session_id,
            exc,
        )
        return "failed"

    thread_id = topic.message_thread_id
    if live_window_id:
        _bind_resume_topic(
            target,
            thread_id,
            live_window_id,
            _window_name_for_live_binding(live_window_id, topic_name),
        )
        return "rebound"

    return await _resume_session_into_topic(
        context,
        target,
        thread_id,
        entry,
        topic_name,
    )


def _window_name_for_live_binding(window_id: str, fallback: str) -> str:
    view = session_manager.view_window(window_id)
    return view.window_name if view and view.window_name else fallback


async def _resume_session_into_topic(
    context: ContextTypes.DEFAULT_TYPE,
    target: ResumeTopicsTarget,
    thread_id: int,
    entry: ResumeEntry,
    topic_name: str,
) -> Literal["created", "failed"]:
    success, message, created_wname, created_wid, _provider_name = (
        await _create_resume_window(
            target.user_id,
            thread_id,
            entry.session_id,
            entry.cwd,
            entry.transcript_path,
            preferred_window_name=topic_name,
        )
    )
    if not success:
        logger.warning(
            "Failed to resume session %s into topic %s: %s",
            entry.session_id,
            thread_id,
            message,
        )
        with contextlib.suppress(TelegramError):
            await context.bot.delete_forum_topic(
                chat_id=target.chat_id,
                message_thread_id=thread_id,
            )
        return "failed"

    _bind_resume_topic(target, thread_id, created_wid, created_wname)
    return "created"


def _bind_resume_topic(
    target: ResumeTopicsTarget,
    thread_id: int,
    window_id: str,
    window_name: str,
) -> None:
    thread_router.bind_thread(
        target.user_id,
        thread_id,
        window_id,
        window_name=window_name,
    )
    thread_router.set_group_chat_id(target.user_id, thread_id, target.chat_id)


def _bound_session_ids(user_id: int) -> set[str]:
    """Return session ids that already have Telegram topic bindings."""
    result: set[str] = set()
    for bound_user_id, _thread_id, window_id in thread_router.iter_thread_bindings():
        if bound_user_id != user_id:
            continue
        view = session_manager.view_window(window_id)
        if view and view.session_id:
            result.add(view.session_id)
    return result


async def _live_unbound_windows_by_session(user_id: int) -> dict[str, str]:
    """Return live, unbound tmux windows keyed by their session id."""
    bound_windows = set(thread_router.get_all_thread_windows(user_id).values())
    result: dict[str, str] = {}
    for window_id in session_manager.iter_window_ids():
        if window_id in bound_windows:
            continue
        view = session_manager.view_window(window_id)
        if not view or not view.session_id:
            continue
        if await tmux_manager.find_window_by_id(window_id):
            result.setdefault(view.session_id, window_id)
    return result


def _topic_name_for_resume(entry: ResumeEntry) -> str:
    """Build a Telegram topic title from Codex's best available chat summary."""
    summary = _clean_title_part(entry.summary) or entry.session_id[:12]
    project = _clean_title_part(Path(entry.cwd).name)
    if project and project.lower() not in summary.lower():
        title = f"{summary} - {project}"
    else:
        title = summary
    return title[:_TELEGRAM_TOPIC_NAME_MAX].rstrip(" -") or "Codex session"


def _clean_title_part(value: str) -> str:
    return " ".join(str(value or "").split())


def _format_resume_topics_reply(
    *,
    created: int,
    rebound: int,
    failed: int,
    already_bound: int,
    missing_cwd: int,
    limited: int,
) -> str:
    total = created + rebound
    if total:
        head = f"\u2705 Created {total} resume topic(s)."
    else:
        head = "\u2139\ufe0f No dormant sessions needed new topics."

    details: list[str] = []
    if created:
        details.append(f"Started {created} resumed Codex session(s).")
    if rebound:
        details.append(f"Bound {rebound} live tmux session(s).")
    if already_bound:
        details.append(f"Skipped {already_bound} already in topics.")
    if missing_cwd:
        details.append(f"Skipped {missing_cwd} with missing folders.")
    if limited:
        details.append(f"Left {limited} older session(s) untouched for now.")
    if failed:
        details.append(f"{failed} failed; check logs before retrying.")

    return "\n".join([head, *details])
