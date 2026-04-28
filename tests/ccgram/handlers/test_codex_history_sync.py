from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from ccgram.handlers.codex_history_sync import (
    CodexHistoryState,
    PendingCodexTopic,
    _read_codex_event_history,
    _load_state,
    _save_state,
    activate_pending_codex_topic,
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
    )


def _bot(thread_id: int = 77) -> MagicMock:
    topic = MagicMock()
    topic.message_thread_id = thread_id
    bot = MagicMock()
    bot.create_forum_topic = AsyncMock(return_value=topic)
    return bot


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
