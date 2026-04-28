from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.handlers.resume_command import ResumeEntry
from ccgram.handlers.resume_topics_command import resume_topics_command

_RT = "ccgram.handlers.resume_topics_command"


def _make_update(
    *,
    chat_id: int = -100999,
    user_id: int = 100,
) -> MagicMock:
    chat = MagicMock()
    chat.id = chat_id
    chat.type = "supergroup"
    chat.is_forum = True

    msg = MagicMock()
    msg.chat = chat

    update = MagicMock()
    update.effective_user = MagicMock(id=user_id)
    update.effective_chat = chat
    update.message = msg
    return update


def _make_context(*, thread_id: int = 77) -> MagicMock:
    topic = MagicMock()
    topic.message_thread_id = thread_id

    ctx = MagicMock()
    ctx.bot = MagicMock()
    ctx.bot.create_forum_topic = AsyncMock(return_value=topic)
    ctx.bot.delete_forum_topic = AsyncMock()
    return ctx


def _mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.capabilities.name = "codex"
    provider.capabilities.supports_resume = True
    return provider


async def test_resume_topics_creates_topic_and_resumes_session(tmp_path) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    entry = ResumeEntry(
        "sess-1",
        "reply hello",
        str(cwd),
        "/tmp/session.jsonl",
    )
    update = _make_update()
    ctx = _make_context(thread_id=77)

    with (
        patch(f"{_RT}.config") as mock_config,
        patch(f"{_RT}.get_provider", return_value=_mock_provider()),
        patch(f"{_RT}.scan_all_sessions", return_value=[entry]),
        patch(f"{_RT}.session_manager") as mock_sm,
        patch(f"{_RT}.thread_router") as mock_tr,
        patch(f"{_RT}._create_resume_window", new=AsyncMock()) as mock_create,
        patch(f"{_RT}.safe_reply", new=AsyncMock()) as mock_safe_reply,
    ):
        mock_config.is_user_allowed.return_value = True
        mock_tr.iter_thread_bindings.return_value = []
        mock_tr.get_all_thread_windows.return_value = {}
        mock_sm.iter_window_ids.return_value = []
        mock_create.return_value = (
            True,
            "ok",
            "reply hello - second-brain",
            "@5",
            "codex",
        )

        await resume_topics_command(update, ctx)

    ctx.bot.create_forum_topic.assert_awaited_once_with(
        chat_id=-100999,
        name="reply hello - second-brain",
    )
    mock_create.assert_awaited_once_with(
        100,
        77,
        "sess-1",
        str(cwd),
        "/tmp/session.jsonl",
        preferred_window_name="reply hello - second-brain",
    )
    mock_tr.bind_thread.assert_called_once_with(
        100,
        77,
        "@5",
        window_name="reply hello - second-brain",
    )
    mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100999)
    assert "Created 1 resume topic" in mock_safe_reply.call_args.args[1]


async def test_resume_topics_skips_sessions_already_bound(tmp_path) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    entry = ResumeEntry("sess-1", "reply hello", str(cwd), "/tmp/session.jsonl")
    update = _make_update()
    ctx = _make_context()

    bound_view = MagicMock()
    bound_view.session_id = "sess-1"

    with (
        patch(f"{_RT}.config") as mock_config,
        patch(f"{_RT}.get_provider", return_value=_mock_provider()),
        patch(f"{_RT}.scan_all_sessions", return_value=[entry]),
        patch(f"{_RT}.session_manager") as mock_sm,
        patch(f"{_RT}.thread_router") as mock_tr,
        patch(f"{_RT}._create_resume_window", new=AsyncMock()) as mock_create,
        patch(f"{_RT}.safe_reply", new=AsyncMock()) as mock_safe_reply,
    ):
        mock_config.is_user_allowed.return_value = True
        mock_tr.iter_thread_bindings.return_value = [(100, 42, "@1")]
        mock_sm.view_window.return_value = bound_view

        await resume_topics_command(update, ctx)

    ctx.bot.create_forum_topic.assert_not_awaited()
    mock_create.assert_not_awaited()
    assert "already in topics" in mock_safe_reply.call_args.args[1]


async def test_resume_topics_binds_live_unbound_window_without_resuming(tmp_path) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    entry = ResumeEntry("sess-live", "existing chat", str(cwd), "/tmp/session.jsonl")
    update = _make_update()
    ctx = _make_context(thread_id=88)

    live_view = MagicMock()
    live_view.session_id = "sess-live"
    live_view.window_name = "desktop-codex"

    with (
        patch(f"{_RT}.config") as mock_config,
        patch(f"{_RT}.get_provider", return_value=_mock_provider()),
        patch(f"{_RT}.scan_all_sessions", return_value=[entry]),
        patch(f"{_RT}.session_manager") as mock_sm,
        patch(f"{_RT}.thread_router") as mock_tr,
        patch(f"{_RT}.tmux_manager") as mock_tmux,
        patch(f"{_RT}._create_resume_window", new=AsyncMock()) as mock_create,
        patch(f"{_RT}.safe_reply", new=AsyncMock()),
    ):
        mock_config.is_user_allowed.return_value = True
        mock_tr.iter_thread_bindings.return_value = []
        mock_tr.get_all_thread_windows.return_value = {}
        mock_sm.iter_window_ids.return_value = ["@3"]
        mock_sm.view_window.return_value = live_view
        mock_tmux.find_window_by_id = AsyncMock(return_value=MagicMock())

        await resume_topics_command(update, ctx)

    ctx.bot.create_forum_topic.assert_awaited_once_with(
        chat_id=-100999,
        name="existing chat - second-brain",
    )
    mock_create.assert_not_awaited()
    mock_tr.bind_thread.assert_called_once_with(
        100,
        88,
        "@3",
        window_name="desktop-codex",
    )
