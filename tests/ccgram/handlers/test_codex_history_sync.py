from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.codex_app_server import (
    CodexAppServerBusyError,
    CodexAppServerError,
    CodexTurnSubmission,
)
from ccgram.handlers.codex_history_sync import (
    CodexHistoryState,
    PendingCodexTopic,
    _read_codex_event_history,
    _load_state,
    _save_state,
    activate_pending_codex_topic,
    rename_app_server_synced_topic,
    resolve_pending_codex_attachment_target,
    submit_attachment_to_pending_codex_topic,
    submit_or_activate_pending_codex_topic,
    sync_codex_history_once,
)
from ccgram.handlers.resume_command import ResumeEntry

_CHS = "ccgram.handlers.codex_history_sync"


def _config(tmp_path):
    return SimpleNamespace(
        allowed_users={100},
        group_id=-100999,
        codex_history_sync_enabled=True,
        codex_history_sync_interval=5.0,
        codex_history_sync_file=tmp_path / "codex_history_topics.json",
        codex_app_server_enabled=True,
        codex_app_server_url="ws://127.0.0.1:9234",
        codex_app_server_timeout=2.0,
    )


def _bot(thread_id: int = 77) -> MagicMock:
    topic = MagicMock()
    topic.message_thread_id = thread_id
    bot = MagicMock()
    bot.create_forum_topic = AsyncMock(return_value=topic)
    return bot


@pytest.fixture(autouse=True)
def mock_read_thread_names():
    with patch(
        f"{_CHS}.read_thread_names_from_app_server",
        new=AsyncMock(return_value={}),
    ) as mock:
        yield mock


def _write_transcript(path, *payloads: str) -> None:
    path.write_text("\n".join(payloads) + "\n")


async def test_first_scan_seeds_existing_sessions_without_creating_topics(
    tmp_path,
) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    entry = ResumeEntry("sess-1", "reply hello", str(cwd), "/tmp/session.jsonl")
    bot = _bot()

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.scan_all_sessions", return_value=[entry]),
    ):
        await sync_codex_history_once(bot)
        state = _load_state()

    bot.create_forum_topic.assert_not_awaited()
    assert state.initialized is True
    assert state.seen_session_ids == {"sess-1"}
    assert state.pending_topics == {}


async def test_scan_creates_lazy_topic_for_new_codex_session(tmp_path) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    transcript = tmp_path / "new.jsonl"
    _write_transcript(
        transcript,
        '{"type":"event_msg","payload":{"type":"user_message","message":"Add mobile bridge"}}',
    )
    entry = ResumeEntry("sess-new", "Add mobile bridge", str(cwd), str(transcript))
    bot = _bot(thread_id=88)

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}._bound_session_ids", return_value=set()),
        patch(f"{_CHS}.scan_all_sessions", return_value=[entry]),
        patch(f"{_CHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        mock_safe_send.return_value = MagicMock()
        _save_state(CodexHistoryState(True, set(), {}))
        await sync_codex_history_once(bot)
        state = _load_state()

    bot.create_forum_topic.assert_awaited_once_with(
        chat_id=-100999,
        name="Add mobile bridge - second-brain",
    )
    assert state.seen_session_ids == {"sess-new"}
    pending = state.pending_topics["100:88"]
    assert pending.session_id == "sess-new"
    assert pending.thread_id == 88
    assert pending.history_offset > 0
    mock_safe_send.assert_awaited_once()
    assert "Add mobile bridge" in mock_safe_send.call_args.args[2]


async def test_new_topic_state_saved_before_title_sync(tmp_path) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    transcript = tmp_path / "new.jsonl"
    _write_transcript(
        transcript,
        '{"type":"event_msg","payload":{"type":"user_message","message":"Add mobile bridge"}}',
    )
    entry = ResumeEntry("sess-new", "Add mobile bridge", str(cwd), str(transcript))
    bot = _bot(thread_id=88)

    async def assert_state_saved(*_args) -> None:
        state = _load_state()
        assert "100:88" in state.pending_topics
        assert state.seen_session_ids == {"sess-new"}

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}._bound_session_ids", return_value=set()),
        patch(f"{_CHS}.scan_all_sessions", return_value=[entry]),
        patch(f"{_CHS}.safe_send", new=AsyncMock()) as mock_safe_send,
        patch(
            f"{_CHS}._sync_app_server_thread_names",
            new=AsyncMock(side_effect=assert_state_saved),
        ),
    ):
        mock_safe_send.return_value = MagicMock()
        _save_state(CodexHistoryState(True, set(), {}))
        await sync_codex_history_once(bot)

    bot.create_forum_topic.assert_awaited_once()


async def test_scan_marks_bound_sessions_seen_without_duplicate_topic(tmp_path) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    entry = ResumeEntry("sess-bound", "Already open", str(cwd), "/tmp/bound.jsonl")
    bot = _bot()

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}._bound_session_ids", return_value={"sess-bound"}),
        patch(f"{_CHS}.scan_all_sessions", return_value=[entry]),
    ):
        _save_state(CodexHistoryState(True, set(), {}))
        await sync_codex_history_once(bot)
        state = _load_state()

    bot.create_forum_topic.assert_not_awaited()
    assert state.seen_session_ids == {"sess-bound"}


async def test_scan_records_existing_bound_codex_topic_for_app_server_sync(
    tmp_path,
) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    transcript = tmp_path / "bound.jsonl"
    _write_transcript(
        transcript,
        '{"type":"event_msg","payload":{"type":"user_message","message":"old chat"}}',
    )
    entry = ResumeEntry("sess-bound", "old chat", str(cwd), str(transcript))
    bot = _bot()
    view = SimpleNamespace(
        session_id="sess-bound",
        provider_name="codex",
        transcript_path=transcript,
        cwd=str(cwd),
    )

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.thread_router") as mock_tr,
        patch(f"{_CHS}.session_manager") as mock_sm,
        patch(f"{_CHS}._bound_session_ids", return_value={"sess-bound"}),
        patch(f"{_CHS}.scan_all_sessions", return_value=[entry]),
    ):
        mock_tr.iter_thread_bindings.return_value = [(100, 166, "@11")]
        mock_tr.resolve_chat_id.return_value = -100999
        mock_tr.get_display_name.return_value = "old desktop thread"
        mock_sm.view_window.return_value = view
        _save_state(CodexHistoryState(True, set(), {}))
        await sync_codex_history_once(bot)
        state = _load_state()

    bot.create_forum_topic.assert_not_awaited()
    pending = state.pending_topics["100:166"]
    assert pending.session_id == "sess-bound"
    assert pending.thread_id == 166
    assert pending.app_server_thread_id == "sess-bound"
    assert pending.history_offset == transcript.stat().st_size
    assert state.seen_session_ids == {"sess-bound"}


async def test_sync_app_server_thread_name_updates_telegram_topic_and_state(
    tmp_path,
    mock_read_thread_names: AsyncMock,
) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    entry = ResumeEntry("sess-1", "Existing session", str(cwd), str(transcript))
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="Existing session",
        cwd=str(cwd),
        transcript_path=str(transcript),
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="Old title",
        created_at=1.0,
        app_server_thread_id="thread-1",
    )
    bot = _bot()
    mock_read_thread_names.return_value = {"thread-1": "Desktop Title"}

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}._bound_session_ids", return_value=set()),
        patch(f"{_CHS}.scan_all_sessions", return_value=[entry]),
        patch(f"{_CHS}.thread_router") as mock_tr,
        patch(f"{_CHS}.tmux_manager") as mock_tm,
        patch(f"{_CHS}.session_manager") as mock_sm,
        patch(f"{_CHS}.sync_topic_name", new=AsyncMock()) as mock_sync_topic_name,
    ):
        mock_tr.iter_thread_bindings.return_value = []
        mock_tr.get_window_for_thread.return_value = "@11"
        mock_tr.get_display_name.return_value = "Old title"
        mock_tm.rename_window = AsyncMock(return_value=True)
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))
        await sync_codex_history_once(bot)
        state = _load_state()

    mock_read_thread_names.assert_awaited_once_with(
        "ws://127.0.0.1:9234",
        {"thread-1"},
        timeout=2.0,
    )
    mock_sync_topic_name.assert_awaited_once_with(
        bot,
        -100999,
        77,
        "Desktop Title",
    )
    mock_tm.rename_window.assert_awaited_once_with("@11", "Desktop Title")
    mock_sm.set_display_name.assert_called_once_with("@11", "Desktop Title")
    assert state.pending_topics["100:77"].topic_name == "Desktop Title"


async def test_pending_topic_backfills_later_agent_message(tmp_path) -> None:
    transcript = tmp_path / "session.jsonl"
    user_line = (
        '{"type":"event_msg","payload":{"type":"user_message",'
        '"message":"telegram new topic"}}'
    )
    agent_line = (
        '{"type":"event_msg","payload":{"type":"agent_message",'
        '"message":"hello world"}}'
    )
    _write_transcript(transcript, user_line)
    first_offset = transcript.stat().st_size
    _write_transcript(transcript, user_line, agent_line)

    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="telegram new topic",
        cwd="/tmp/project",
        transcript_path=str(transcript),
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="telegram new topic - project",
        created_at=1.0,
        history_offset=first_offset,
    )
    bot = _bot()

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.scan_all_sessions", return_value=[]),
        patch(f"{_CHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        mock_safe_send.return_value = MagicMock()
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))
        await sync_codex_history_once(bot)
        state = _load_state()

    mock_safe_send.assert_awaited_once()
    assert mock_safe_send.call_args.args[2] == "hello world"
    assert state.pending_topics["100:77"].history_offset == transcript.stat().st_size


async def test_pending_history_offset_survives_title_sync_timeout(
    tmp_path,
    mock_read_thread_names: AsyncMock,
) -> None:
    transcript = tmp_path / "session.jsonl"
    user_line = (
        '{"type":"event_msg","payload":{"type":"user_message",'
        '"message":"telegram new topic"}}'
    )
    agent_line = (
        '{"type":"event_msg","payload":{"type":"agent_message",'
        '"message":"hello world"}}'
    )
    _write_transcript(transcript, user_line)
    first_offset = transcript.stat().st_size
    _write_transcript(transcript, user_line, agent_line)
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="telegram new topic",
        cwd="/tmp/project",
        transcript_path=str(transcript),
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="telegram new topic - project",
        created_at=1.0,
        history_offset=first_offset,
        app_server_thread_id="thread-1",
    )
    bot = _bot()
    mock_read_thread_names.side_effect = TimeoutError

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.scan_all_sessions", return_value=[]),
        patch(f"{_CHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        mock_safe_send.return_value = MagicMock()
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))
        await sync_codex_history_once(bot)
        state = _load_state()
        await sync_codex_history_once(bot)

    mock_safe_send.assert_awaited_once()
    assert mock_safe_send.call_args.args[2] == "hello world"
    assert state.pending_topics["100:77"].history_offset == transcript.stat().st_size


async def test_pending_topic_skips_app_server_submitted_user_echo(tmp_path) -> None:
    transcript = tmp_path / "session.jsonl"
    first_line = (
        '{"type":"event_msg","payload":{"type":"user_message",'
        '"message":"first prompt"}}'
    )
    submitted_line = (
        '{"type":"event_msg","payload":{"type":"user_message",'
        '"message":"continue from mobile"}}'
    )
    agent_line = (
        '{"type":"event_msg","payload":{"type":"agent_message",'
        '"message":"mobile answer"}}'
    )
    _write_transcript(transcript, first_line)
    first_offset = transcript.stat().st_size
    _write_transcript(transcript, first_line, submitted_line, agent_line)
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="first prompt",
        cwd="/tmp/project",
        transcript_path=str(transcript),
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="first prompt - project",
        created_at=1.0,
        history_offset=first_offset,
        app_server_thread_id="sess-1",
        submitted_prompt_echoes=("continue from mobile",),
    )
    bot = _bot()

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.scan_all_sessions", return_value=[]),
        patch(f"{_CHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        mock_safe_send.return_value = MagicMock()
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))
        await sync_codex_history_once(bot)
        state = _load_state()

    mock_safe_send.assert_awaited_once()
    assert mock_safe_send.call_args.args[2] == "mobile answer"
    updated = state.pending_topics["100:77"]
    assert updated.history_offset == transcript.stat().st_size
    assert updated.submitted_prompt_echoes == ()


async def test_pending_topic_skips_multiple_submitted_user_echoes(tmp_path) -> None:
    transcript = tmp_path / "session.jsonl"
    first_line = (
        '{"type":"event_msg","payload":{"type":"user_message",'
        '"message":"first prompt"}}'
    )
    submitted_one = (
        '{"type":"event_msg","payload":{"type":"user_message",'
        '"message":"first mobile turn"}}'
    )
    submitted_two = (
        '{"type":"event_msg","payload":{"type":"user_message",'
        '"message":"second mobile turn"}}'
    )
    agent_line = (
        '{"type":"event_msg","payload":{"type":"agent_message",'
        '"message":"combined answer"}}'
    )
    _write_transcript(transcript, first_line)
    first_offset = transcript.stat().st_size
    _write_transcript(transcript, first_line, submitted_one, submitted_two, agent_line)
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="first prompt",
        cwd="/tmp/project",
        transcript_path=str(transcript),
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="first prompt - project",
        created_at=1.0,
        history_offset=first_offset,
        app_server_thread_id="sess-1",
        submitted_prompt_echoes=("first mobile turn", "second mobile turn"),
    )
    bot = _bot()

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.scan_all_sessions", return_value=[]),
        patch(f"{_CHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        mock_safe_send.return_value = MagicMock()
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))
        await sync_codex_history_once(bot)
        state = _load_state()

    mock_safe_send.assert_awaited_once()
    assert mock_safe_send.call_args.args[2] == "combined answer"
    assert state.pending_topics["100:77"].submitted_prompt_echoes == ()


def test_load_state_migrates_legacy_last_submitted_prompt(tmp_path) -> None:
    cfg = _config(tmp_path)
    cfg.codex_history_sync_file.write_text(
        (
            '{"initialized": true, "seen_session_ids": [], "pending_topics": {'
            '"100:77": {'
            '"session_id": "sess-1",'
            '"summary": "hello",'
            '"cwd": "/tmp/project",'
            '"transcript_path": "/tmp/session.jsonl",'
            '"user_id": 100,'
            '"chat_id": -100999,'
            '"thread_id": 77,'
            '"topic_name": "hello - project",'
            '"created_at": 1.0,'
            '"last_submitted_prompt": "legacy prompt"'
            "}}}"
        ),
        encoding="utf-8",
    )

    with patch(f"{_CHS}.config", cfg):
        state = _load_state()

    assert state.pending_topics["100:77"].submitted_prompt_echoes == (
        "legacy prompt",
    )


def test_codex_event_history_reads_only_user_and_agent_messages(tmp_path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        '{"type":"response_item","payload":{"type":"message","role":"user","content":[{"text":"AGENTS"}]}}',
        '{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}',
        '{"type":"event_msg","payload":{"type":"agent_message","message":"world"}}',
    )

    messages, offset = _read_codex_event_history(str(transcript), 0)

    assert messages == ["\U0001f464 hello", "world"]
    assert offset == transcript.stat().st_size


async def test_rename_app_server_synced_topic_pushes_telegram_name_to_app_server(
    tmp_path,
) -> None:
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="reply hello",
        cwd="/tmp/project",
        transcript_path="/tmp/session.jsonl",
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="Old title",
        created_at=1.0,
        app_server_thread_id="thread-1",
    )

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.thread_router") as mock_tr,
        patch(
            f"{_CHS}.set_thread_name_on_app_server",
            new=AsyncMock(),
        ) as mock_set_thread_name,
    ):
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))
        synced = await rename_app_server_synced_topic(
            100,
            77,
            -100999,
            "\U0001f7e2 Telegram Title  ",
        )
        state = _load_state()

    assert synced is True
    mock_set_thread_name.assert_awaited_once_with(
        "ws://127.0.0.1:9234",
        "thread-1",
        "Telegram Title",
        timeout=2.0,
    )
    mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100999)
    assert state.pending_topics["100:77"].topic_name == "Telegram Title"


async def test_activate_pending_topic_resumes_and_binds(tmp_path) -> None:
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="reply hello",
        cwd="/tmp/project",
        transcript_path="/tmp/session.jsonl",
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="reply hello - project",
        created_at=1.0,
    )

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.thread_router") as mock_tr,
        patch(f"{_CHS}._create_resume_window", new=AsyncMock()) as mock_create,
    ):
        _save_state(
            CodexHistoryState(True, {"sess-1"}, {"100:77": pending})
        )
        mock_create.return_value = (
            True,
            "ok",
            "reply hello - project",
            "@5",
            "codex",
        )

        window_id = await activate_pending_codex_topic(100, 77, -100999)
        state = _load_state()

    assert window_id == "@5"
    mock_create.assert_awaited_once_with(
        100,
        77,
        "sess-1",
        "/tmp/project",
        "/tmp/session.jsonl",
        preferred_window_name="reply hello - project",
    )
    mock_tr.bind_thread.assert_called_once_with(
        100,
        77,
        "@5",
        window_name="reply hello - project",
    )
    mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100999)
    assert state.pending_topics == {}


async def test_pending_topic_submits_to_app_server_without_tmux_resume(tmp_path) -> None:
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="reply hello",
        cwd="/tmp/project",
        transcript_path="/tmp/session.jsonl",
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="reply hello - project",
        created_at=1.0,
        app_server_thread_id="thread-1",
    )

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.thread_router") as mock_tr,
        patch(f"{_CHS}._create_resume_window", new=AsyncMock()) as mock_create,
        patch(f"{_CHS}.submit_turn_to_app_server", new=AsyncMock()) as mock_submit,
    ):
        mock_submit.return_value = CodexTurnSubmission("thread-1", "turn-1")
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))

        action = await submit_or_activate_pending_codex_topic(
            100,
            77,
            -100999,
            "continue from mobile",
        )
        state = _load_state()

    assert action.status == "submitted"
    assert action.message == "turn-1"
    mock_submit.assert_awaited_once_with(
        "ws://127.0.0.1:9234",
        "thread-1",
        "continue from mobile",
        timeout=2.0,
    )
    mock_create.assert_not_awaited()
    mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100999)
    updated = state.pending_topics["100:77"]
    assert updated.submitted_prompt_echoes == ("continue from mobile",)
    assert updated.app_server_thread_id == "thread-1"


async def test_resolve_pending_codex_attachment_target(tmp_path) -> None:
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="reply hello",
        cwd="/tmp/project",
        transcript_path="/tmp/session.jsonl",
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="reply hello - project",
        created_at=1.0,
        app_server_thread_id="thread-1",
    )

    with patch(f"{_CHS}.config", _config(tmp_path)):
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))
        target = await resolve_pending_codex_attachment_target(100, 77)

    assert target is not None
    assert target.cwd == "/tmp/project"
    assert target.app_server_thread_id == "thread-1"


async def test_attachment_to_pending_topic_submits_extra_input(tmp_path) -> None:
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="reply hello",
        cwd="/tmp/project",
        transcript_path="/tmp/session.jsonl",
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="reply hello - project",
        created_at=1.0,
        app_server_thread_id="thread-1",
    )
    extra_input = [{"type": "localImage", "path": "/tmp/project/shot.jpg"}]

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.thread_router") as mock_tr,
        patch(f"{_CHS}.submit_turn_to_app_server", new=AsyncMock()) as mock_submit,
    ):
        mock_submit.return_value = CodexTurnSubmission("thread-1", "turn-1")
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))

        action = await submit_attachment_to_pending_codex_topic(
            100,
            77,
            -100999,
            "Please inspect .ccgram-uploads/shot.jpg",
            extra_input=extra_input,
        )
        state = _load_state()

    assert action.status == "submitted"
    mock_submit.assert_awaited_once_with(
        "ws://127.0.0.1:9234",
        "thread-1",
        "Please inspect .ccgram-uploads/shot.jpg",
        timeout=2.0,
        extra_input=extra_input,
    )
    mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100999)
    updated = state.pending_topics["100:77"]
    assert updated.submitted_prompt_echoes == (
        "Please inspect .ccgram-uploads/shot.jpg",
    )


async def test_pending_topic_busy_does_not_fall_back_to_tmux(tmp_path) -> None:
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="reply hello",
        cwd="/tmp/project",
        transcript_path="/tmp/session.jsonl",
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="reply hello - project",
        created_at=1.0,
        app_server_thread_id="thread-1",
    )

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}._create_resume_window", new=AsyncMock()) as mock_create,
        patch(f"{_CHS}.submit_turn_to_app_server", new=AsyncMock()) as mock_submit,
    ):
        mock_submit.side_effect = CodexAppServerBusyError("active")
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))

        action = await submit_or_activate_pending_codex_topic(
            100,
            77,
            -100999,
            "continue from mobile",
        )

    assert action.status == "busy"
    mock_create.assert_not_awaited()


async def test_pending_topic_app_server_failure_falls_back_to_tmux(tmp_path) -> None:
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="reply hello",
        cwd="/tmp/project",
        transcript_path="/tmp/session.jsonl",
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="reply hello - project",
        created_at=1.0,
        app_server_thread_id="thread-1",
    )

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.thread_router") as mock_tr,
        patch(f"{_CHS}._create_resume_window", new=AsyncMock()) as mock_create,
        patch(f"{_CHS}.submit_turn_to_app_server", new=AsyncMock()) as mock_submit,
    ):
        mock_submit.side_effect = CodexAppServerError("offline")
        mock_create.return_value = (
            True,
            "ok",
            "reply hello - project",
            "@5",
            "codex",
        )
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))

        action = await submit_or_activate_pending_codex_topic(
            100,
            77,
            -100999,
            "continue from mobile",
        )
        state = _load_state()

    assert action.status == "fallback_window"
    assert action.window_id == "@5"
    mock_create.assert_awaited_once()
    mock_tr.bind_thread.assert_called_once()
    assert state.pending_topics == {}


async def test_pending_bound_topic_app_server_failure_uses_existing_window(
    tmp_path,
) -> None:
    pending = PendingCodexTopic(
        session_id="sess-1",
        summary="reply hello",
        cwd="/tmp/project",
        transcript_path="/tmp/session.jsonl",
        user_id=100,
        chat_id=-100999,
        thread_id=77,
        topic_name="reply hello - project",
        created_at=1.0,
        app_server_thread_id="thread-1",
    )

    with (
        patch(f"{_CHS}.config", _config(tmp_path)),
        patch(f"{_CHS}.thread_router") as mock_tr,
        patch(f"{_CHS}._create_resume_window", new=AsyncMock()) as mock_create,
        patch(f"{_CHS}.submit_turn_to_app_server", new=AsyncMock()) as mock_submit,
    ):
        mock_submit.side_effect = CodexAppServerError("offline")
        _save_state(CodexHistoryState(True, {"sess-1"}, {"100:77": pending}))

        action = await submit_or_activate_pending_codex_topic(
            100,
            77,
            -100999,
            "continue from mobile",
            fallback_window_id="@11",
        )
        state = _load_state()

    assert action.status == "fallback_window"
    assert action.window_id == "@11"
    mock_create.assert_not_awaited()
    mock_tr.set_group_chat_id.assert_called_once_with(100, 77, -100999)
    assert state.pending_topics["100:77"] == pending
