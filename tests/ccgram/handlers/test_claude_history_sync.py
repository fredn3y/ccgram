import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.claude_history_sync import (
    ClaudeHistoryState,
    PendingClaudeTopic,
    _active_turns,
    _claude_turn_command,
    _load_state,
    _read_claude_history_entries,
    _save_state,
    forget_pending_claude_topic,
    submit_to_pending_claude_topic,
    sync_claude_history_once,
)
from ccgram.handlers.codex_history_sync import _SyncTarget
from ccgram.handlers.resume_command import ResumeEntry

_CLHS = "ccgram.handlers.claude_history_sync"


def _config(tmp_path):
    return SimpleNamespace(
        allowed_users={100},
        group_id=-100999,
        claude_history_sync_enabled=True,
        claude_history_sync_interval=5.0,
        claude_history_sync_file=tmp_path / "claude_history_topics.json",
        claude_history_model="",
        claude_history_turn_timeout=120.0,
    )


def _bot(thread_id: int = 77) -> MagicMock:
    topic = MagicMock()
    topic.message_thread_id = thread_id
    bot = MagicMock()
    bot.create_forum_topic = AsyncMock(return_value=topic)
    return bot


def _target() -> _SyncTarget:
    return _SyncTarget(user_id=100, chat_id=-100999)


@pytest.fixture(autouse=True)
def clear_active_turns():
    _active_turns.clear()
    yield
    _active_turns.clear()


def _user_line(text: str, **extra) -> str:
    return json.dumps(
        {"type": "user", "message": {"role": "user", "content": text}, **extra}
    )


def _assistant_line(*blocks: dict, **extra) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": list(blocks)},
            **extra,
        }
    )


def _write_transcript(path, *payloads: str) -> None:
    path.write_text("\n".join(payloads) + "\n")


def _pending(transcript, thread_id: int = 77, **overrides) -> PendingClaudeTopic:
    fields = {
        "session_id": "sess-1",
        "summary": "hello",
        "cwd": str(transcript.parent),
        "transcript_path": str(transcript),
        "user_id": 100,
        "chat_id": -100999,
        "thread_id": thread_id,
        "topic_name": "hello - proj",
        "created_at": 0.0,
    }
    fields.update(overrides)
    return PendingClaudeTopic(**fields)


async def test_first_scan_seeds_existing_sessions_without_creating_topics(
    tmp_path,
) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    entry = ResumeEntry("sess-1", "reply hello", str(cwd), "/tmp/session.jsonl")
    bot = _bot()

    with (
        patch(f"{_CLHS}.config", _config(tmp_path)),
        patch(f"{_CLHS}._default_target", return_value=_target()),
        patch(f"{_CLHS}.scan_all_sessions", return_value=[entry]),
    ):
        await sync_claude_history_once(bot)
        state = _load_state()

    bot.create_forum_topic.assert_not_awaited()
    assert state.initialized is True
    assert state.seen_session_ids == {"sess-1"}
    assert state.pending_topics == {}


async def test_scan_creates_lazy_topic_and_backfills_history(tmp_path) -> None:
    cwd = tmp_path / "second-brain"
    cwd.mkdir()
    transcript = tmp_path / "sess-new.jsonl"
    _write_transcript(
        transcript,
        json.dumps({"type": "queue-operation", "operation": "enqueue"}),
        _user_line("Add mobile bridge"),
        _assistant_line({"type": "thinking", "thinking": "hmm"}),
        _assistant_line({"type": "text", "text": "On it."}),
    )
    entry = ResumeEntry("sess-new", "Add mobile bridge", str(cwd), str(transcript))
    bot = _bot(thread_id=88)

    with (
        patch(f"{_CLHS}.config", _config(tmp_path)),
        patch(f"{_CLHS}._default_target", return_value=_target()),
        patch(f"{_CLHS}._bound_session_ids", return_value=set()),
        patch(f"{_CLHS}.scan_all_sessions", return_value=[entry]),
        patch(f"{_CLHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        _save_state(ClaudeHistoryState(True, set(), {}))
        await sync_claude_history_once(bot)
        state = _load_state()

    bot.create_forum_topic.assert_awaited_once()
    key = "100:88"
    assert key in state.pending_topics
    pending = state.pending_topics[key]
    assert pending.session_id == "sess-new"
    assert pending.history_offset == transcript.stat().st_size
    sent_texts = [call.args[2] for call in mock_safe_send.await_args_list]
    assert any("Add mobile bridge" in text for text in sent_texts)
    assert any("On it." in text for text in sent_texts)


async def test_backfill_skips_meta_tool_results_and_commands(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    _write_transcript(
        transcript,
        _user_line("<command-name>/clear</command-name>"),
        _user_line("noise", isMeta=True),
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "content": "ok"}],
                },
            }
        ),
        _user_line("real question"),
        _assistant_line({"type": "tool_use", "name": "Bash", "input": {}}),
        _assistant_line({"type": "text", "text": "real answer"}),
    )

    entries, offset = _read_claude_history_entries(str(transcript), 0)

    assert offset == transcript.stat().st_size
    assert [(entry.role, entry.text) for entry in entries] == [
        ("user", "real question"),
        ("agent", "real answer"),
    ]
    assert entries[0].formatted.startswith("\U0001f464 ")


async def test_ai_title_renames_topic(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    _write_transcript(
        transcript,
        json.dumps({"type": "ai-title", "aiTitle": "Fix login flow"}),
    )
    pending = _pending(transcript)
    state = ClaudeHistoryState(True, {"sess-1"}, {"100:77": pending})

    with (
        patch(f"{_CLHS}.config", _config(tmp_path)),
        patch(f"{_CLHS}._default_target", return_value=_target()),
        patch(f"{_CLHS}.scan_all_sessions", return_value=[]),
        patch(f"{_CLHS}.sync_topic_name", new=AsyncMock()) as mock_sync_name,
        patch(f"{_CLHS}.update_stored_topic_name") as mock_update_stored,
    ):
        _save_state(state)
        await sync_claude_history_once(_bot())
        saved = _load_state()

    mock_sync_name.assert_awaited_once()
    assert mock_sync_name.await_args.args[1:] == (-100999, 77, "Fix login flow")
    mock_update_stored.assert_called_once_with(-100999, 77, "Fix login flow")
    assert saved.pending_topics["100:77"].topic_name == "Fix login flow"


async def test_submitted_prompt_echo_is_suppressed(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    _write_transcript(
        transcript,
        _user_line("do the thing"),
        _assistant_line({"type": "text", "text": "done"}),
    )
    pending = _pending(transcript, submitted_prompt_echoes=("do the thing",))
    state = ClaudeHistoryState(True, {"sess-1"}, {"100:77": pending})

    with (
        patch(f"{_CLHS}.config", _config(tmp_path)),
        patch(f"{_CLHS}._default_target", return_value=_target()),
        patch(f"{_CLHS}.scan_all_sessions", return_value=[]),
        patch(f"{_CLHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        _save_state(state)
        await sync_claude_history_once(_bot())
        saved = _load_state()

    sent_texts = [call.args[2] for call in mock_safe_send.await_args_list]
    assert sent_texts == ["done"]
    assert saved.pending_topics["100:77"].submitted_prompt_echoes == ()


def _fake_process(returncode: int = 0, stdout: bytes = b"{}", stderr: bytes = b""):
    process = SimpleNamespace(
        returncode=returncode,
        communicate=AsyncMock(return_value=(stdout, stderr)),
        kill=MagicMock(),
    )
    return process


async def test_submit_runs_headless_resume_turn(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    transcript.write_text("")
    pending = _pending(transcript)
    process = _fake_process(stdout=b'{"is_error": false}')

    with (
        patch(f"{_CLHS}.config", _config(tmp_path)),
        patch(f"{_CLHS}.thread_router") as mock_router,
        patch(f"{_CLHS}.resolve_launch_command", return_value="claude"),
        patch(
            f"{_CLHS}.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ) as mock_exec,
        patch(f"{_CLHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        _save_state(ClaudeHistoryState(True, {"sess-1"}, {"100:77": pending}))
        action = await submit_to_pending_claude_topic(
            100, 77, -100999, "carry on", _bot()
        )
        turn_task = _active_turns.get("100:77")
        assert turn_task is not None
        await turn_task
        saved = _load_state()

    assert action.status == "submitted"
    mock_router.set_group_chat_id.assert_called_once_with(100, 77, -100999)
    cmd = mock_exec.await_args.args
    assert cmd[0] == "claude"
    assert "--resume" in cmd and "sess-1" in cmd
    assert cmd[-1] == "carry on"
    assert mock_exec.await_args.kwargs["cwd"] == str(transcript.parent)
    assert saved.pending_topics["100:77"].submitted_prompt_echoes == ("carry on",)
    mock_safe_send.assert_not_awaited()


async def test_submit_busy_while_turn_in_flight(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    transcript.write_text("")
    pending = _pending(transcript)
    blocker = asyncio.create_task(asyncio.sleep(30))
    _active_turns["100:77"] = blocker

    try:
        with patch(f"{_CLHS}.config", _config(tmp_path)):
            _save_state(ClaudeHistoryState(True, {"sess-1"}, {"100:77": pending}))
            action = await submit_to_pending_claude_topic(
                100, 77, -100999, "another", _bot()
            )
    finally:
        blocker.cancel()

    assert action.status == "busy"


async def test_submit_not_pending_for_unknown_topic(tmp_path) -> None:
    with patch(f"{_CLHS}.config", _config(tmp_path)):
        action = await submit_to_pending_claude_topic(100, 12, -100999, "hi", _bot())

    assert action.status == "not_pending"


async def test_failed_turn_reports_error_into_topic(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    transcript.write_text("")
    pending = _pending(transcript)
    process = _fake_process(returncode=1, stderr=b"boom: no credentials")

    with (
        patch(f"{_CLHS}.config", _config(tmp_path)),
        patch(f"{_CLHS}.thread_router"),
        patch(f"{_CLHS}.resolve_launch_command", return_value="claude"),
        patch(
            f"{_CLHS}.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ),
        patch(f"{_CLHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        _save_state(ClaudeHistoryState(True, {"sess-1"}, {"100:77": pending}))
        action = await submit_to_pending_claude_topic(
            100, 77, -100999, "carry on", _bot()
        )
        await _active_turns["100:77"]

    assert action.status == "submitted"
    mock_safe_send.assert_awaited_once()
    error_text = mock_safe_send.await_args.args[2]
    assert "exit 1" in error_text
    assert "no credentials" in error_text


async def test_error_result_payload_reports_into_topic(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    transcript.write_text("")
    pending = _pending(transcript)
    payload = json.dumps({"is_error": True, "result": "rate limited"}).encode()
    process = _fake_process(returncode=0, stdout=payload)

    with (
        patch(f"{_CLHS}.config", _config(tmp_path)),
        patch(f"{_CLHS}.thread_router"),
        patch(f"{_CLHS}.resolve_launch_command", return_value="claude"),
        patch(
            f"{_CLHS}.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ),
        patch(f"{_CLHS}.safe_send", new=AsyncMock()) as mock_safe_send,
    ):
        _save_state(ClaudeHistoryState(True, {"sess-1"}, {"100:77": pending}))
        await submit_to_pending_claude_topic(100, 77, -100999, "carry on", _bot())
        await _active_turns["100:77"]

    mock_safe_send.assert_awaited_once()
    assert "rate limited" in mock_safe_send.await_args.args[2]


async def test_submit_fails_when_cwd_missing(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    transcript.write_text("")
    pending = _pending(transcript, cwd=str(tmp_path / "gone"))

    with patch(f"{_CLHS}.config", _config(tmp_path)):
        _save_state(ClaudeHistoryState(True, {"sess-1"}, {"100:77": pending}))
        action = await submit_to_pending_claude_topic(
            100, 77, -100999, "carry on", _bot()
        )

    assert action.status == "failed"
    assert "gone" in action.message


async def test_forget_pending_claude_topic_drops_state(tmp_path) -> None:
    transcript = tmp_path / "sess-1.jsonl"
    transcript.write_text("")
    pending = _pending(transcript)

    with patch(f"{_CLHS}.config", _config(tmp_path)):
        _save_state(ClaudeHistoryState(True, {"sess-1"}, {"100:77": pending}))
        await forget_pending_claude_topic(100, 77)
        saved = _load_state()

    assert saved.pending_topics == {}
    assert saved.seen_session_ids == {"sess-1"}


def test_claude_turn_command_includes_model_when_configured(tmp_path) -> None:
    cfg = _config(tmp_path)
    cfg.claude_history_model = "claude-fable-5"

    with (
        patch(f"{_CLHS}.config", cfg),
        patch(f"{_CLHS}.resolve_launch_command", return_value="claude"),
    ):
        cmd = _claude_turn_command("sess-1", "hello")

    assert cmd[:2] == ["claude", "-p"]
    assert cmd[-3:-1] == ["--model", "claude-fable-5"]
    assert cmd[-1] == "hello"
    assert "--permission-mode" in cmd and "bypassPermissions" in cmd
