import json

import pytest

from ccgram.codex_app_server import (
    CodexAppServerBusyError,
    CodexAppServerClient,
    read_thread_names_from_app_server,
)


class FakeWebSocket:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def recv(self) -> str:
        return self.responses.pop(0)

    async def close(self) -> None:
        self.closed = True


async def test_submit_turn_resumes_not_loaded_thread_before_turn_start(
    monkeypatch,
) -> None:
    socket = FakeWebSocket(
        [
            {"id": 1, "result": {"serverInfo": {"name": "codex"}}},
            {
                "id": 2,
                "result": {
                    "data": [
                        {
                            "id": "thread-1",
                            "status": {"type": "notLoaded"},
                        }
                    ]
                },
            },
            {
                "id": 3,
                "result": {
                    "thread": {
                        "id": "thread-1",
                        "status": {"type": "idle"},
                    }
                },
            },
            {
                "id": 4,
                "result": {
                    "turn": {
                        "id": "turn-1",
                        "status": {"type": "running"},
                    }
                },
            },
        ]
    )

    async def fake_connect(*_args, **_kwargs) -> FakeWebSocket:
        return socket

    monkeypatch.setattr("ccgram.codex_app_server.connect", fake_connect)

    async with CodexAppServerClient("ws://127.0.0.1:9234") as client:
        submission = await client.submit_turn("thread-1", "hello from Telegram")

    assert submission.turn_id == "turn-1"
    assert socket.closed is True
    methods = [sent["method"] for sent in socket.sent]
    assert methods == [
        "initialize",
        "initialized",
        "thread/list",
        "thread/resume",
        "turn/start",
    ]
    turn_start = socket.sent[-1]
    assert turn_start["params"]["threadId"] == "thread-1"
    assert turn_start["params"]["input"] == [
        {"type": "text", "text": "hello from Telegram", "text_elements": []}
    ]


async def test_submit_turn_reports_active_thread_as_busy(monkeypatch) -> None:
    socket = FakeWebSocket(
        [
            {"id": 1, "result": {"serverInfo": {"name": "codex"}}},
            {
                "id": 2,
                "result": {
                    "data": [
                        {
                            "id": "thread-1",
                            "status": {"type": "active"},
                        }
                    ]
                },
            },
        ]
    )

    async def fake_connect(*_args, **_kwargs) -> FakeWebSocket:
        return socket

    monkeypatch.setattr("ccgram.codex_app_server.connect", fake_connect)

    async with CodexAppServerClient("ws://127.0.0.1:9234") as client:
        with pytest.raises(CodexAppServerBusyError):
            await client.submit_turn("thread-1", "hello from Telegram")

    methods = [sent["method"] for sent in socket.sent]
    assert methods == ["initialize", "initialized", "thread/list"]


async def test_overloaded_request_is_retried(monkeypatch) -> None:
    socket = FakeWebSocket(
        [
            {"id": 1, "result": {"serverInfo": {"name": "codex"}}},
            {
                "id": 2,
                "error": {
                    "code": -32001,
                    "message": "Server overloaded",
                },
            },
            {"id": 3, "result": {"data": []}},
        ]
    )

    async def fake_connect(*_args, **_kwargs) -> FakeWebSocket:
        return socket

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ccgram.codex_app_server.connect", fake_connect)
    monkeypatch.setattr("ccgram.codex_app_server.asyncio.sleep", fake_sleep)

    async with CodexAppServerClient("ws://127.0.0.1:9234") as client:
        assert await client.list_threads() == []

    methods = [sent["method"] for sent in socket.sent]
    assert methods == ["initialize", "initialized", "thread/list", "thread/list"]


async def test_set_thread_name_sends_thread_name_set(monkeypatch) -> None:
    socket = FakeWebSocket(
        [
            {"id": 1, "result": {"serverInfo": {"name": "codex"}}},
            {"id": 2, "result": {}},
        ]
    )

    async def fake_connect(*_args, **_kwargs) -> FakeWebSocket:
        return socket

    monkeypatch.setattr("ccgram.codex_app_server.connect", fake_connect)

    async with CodexAppServerClient("ws://127.0.0.1:9234") as client:
        await client.set_thread_name("thread-1", "New title")

    methods = [sent["method"] for sent in socket.sent]
    assert methods == ["initialize", "initialized", "thread/name/set"]
    assert socket.sent[-1]["params"] == {
        "threadId": "thread-1",
        "name": "New title",
    }


async def test_start_turn_can_include_local_image_input(monkeypatch) -> None:
    socket = FakeWebSocket(
        [
            {"id": 1, "result": {"serverInfo": {"name": "codex"}}},
            {
                "id": 2,
                "result": {
                    "turn": {
                        "id": "turn-1",
                    }
                },
            },
        ]
    )

    async def fake_connect(*_args, **_kwargs) -> FakeWebSocket:
        return socket

    monkeypatch.setattr("ccgram.codex_app_server.connect", fake_connect)

    async with CodexAppServerClient("ws://127.0.0.1:9234") as client:
        await client.start_turn(
            "thread-1",
            "Please inspect this screenshot.",
            extra_input=[{"type": "localImage", "path": "/tmp/shot.jpg"}],
        )

    assert socket.sent[-1]["method"] == "turn/start"
    assert socket.sent[-1]["params"]["input"] == [
        {
            "type": "text",
            "text": "Please inspect this screenshot.",
            "text_elements": [],
        },
        {"type": "localImage", "path": "/tmp/shot.jpg"},
    ]


async def test_read_thread_names_reads_list_and_missing_threads(monkeypatch) -> None:
    socket = FakeWebSocket(
        [
            {"id": 1, "result": {"serverInfo": {"name": "codex"}}},
            {
                "id": 2,
                "result": {
                    "data": [
                        {"id": "thread-1", "name": "Listed"},
                        {"id": "other", "name": "Ignore me"},
                    ],
                },
            },
            {"id": 3, "result": {"thread": {"id": "thread-2", "name": "Read"}}},
        ]
    )

    async def fake_connect(*_args, **_kwargs) -> FakeWebSocket:
        return socket

    monkeypatch.setattr("ccgram.codex_app_server.connect", fake_connect)

    names = await read_thread_names_from_app_server(
        "ws://127.0.0.1:9234",
        {"thread-1", "thread-2"},
    )

    assert names == {"thread-1": "Listed", "thread-2": "Read"}
    methods = [sent["method"] for sent in socket.sent]
    assert methods == ["initialize", "initialized", "thread/list", "thread/read"]
