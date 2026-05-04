"""Codex app-server JSON-RPC client.

This is intentionally small and scoped to the bridge methods CCGram needs:
initialize, thread/list, thread/read, thread/resume, thread/name/set, and turn/start.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect, unix_connect
from websockets.exceptions import WebSocketException

from . import __version__

_OVERLOADED_ERROR_CODE = -32001
_OVERLOADED_RETRIES = 3
_DEFAULT_UNIX_SOCKET_RELATIVE = "app-server-control/app-server-control.sock"
_UNIX_SOCKET_WEBSOCKET_URI = "ws://localhost/rpc"


class CodexAppServerError(Exception):
    """Base app-server bridge error."""


class CodexAppServerUnavailableError(CodexAppServerError):
    """The app-server transport could not be reached."""


class CodexAppServerProtocolError(CodexAppServerError):
    """The app-server returned malformed JSON-RPC or an unexpected response."""


class CodexAppServerRequestError(CodexAppServerError):
    """The app-server returned a JSON-RPC error response."""

    def __init__(self, code: int | None, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class CodexAppServerBusyError(CodexAppServerError):
    """The target thread is already running a turn."""


@dataclass(frozen=True, slots=True)
class CodexTurnSubmission:
    """Result from a successful app-server turn submission."""

    thread_id: str
    turn_id: str


class CodexAppServerClient:
    """Tiny JSON-RPC client for Codex app-server websocket transports."""

    def __init__(self, url: str, *, timeout: float = 3.0) -> None:
        self.url = url
        self.timeout = timeout
        self._next_id = 1
        self._ws: Any = None

    async def __aenter__(self) -> CodexAppServerClient:
        try:
            self._ws = await asyncio.wait_for(
                self._connect(),
                timeout=self.timeout,
            )
        except (OSError, TimeoutError, ValueError, WebSocketException) as exc:
            raise CodexAppServerUnavailableError(str(exc)) from exc

        await self._initialize()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def _connect(self) -> Any:
        if self.url.startswith("unix://"):
            return await unix_connect(
                _unix_socket_path_from_url(self.url),
                uri=_UNIX_SOCKET_WEBSOCKET_URI,
                open_timeout=self.timeout,
                close_timeout=self.timeout,
                compression=None,
                user_agent_header=None,
            )
        return await connect(
            self.url,
            open_timeout=self.timeout,
            close_timeout=self.timeout,
        )

    async def _initialize(self) -> None:
        await self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "ccgram",
                    "title": "CCGram",
                    "version": __version__,
                },
                "capabilities": {
                    "experimentalApi": True,
                    "optOutNotificationMethods": [
                        "item/agentMessage/delta",
                        "item/reasoning/textDelta",
                        "item/reasoning/summaryTextDelta",
                    ],
                },
            },
        )
        await self._notification("initialized")

    async def _notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        await self._send(payload)

    async def _request(self, method: str, params: dict[str, Any] | None) -> Any:
        for attempt in range(_OVERLOADED_RETRIES + 1):
            request_id = self._next_id
            self._next_id += 1
            payload: dict[str, Any] = {"id": request_id, "method": method}
            if params is not None:
                payload["params"] = params

            await self._send(payload)
            try:
                return await self._recv_response(request_id)
            except CodexAppServerRequestError as exc:
                if exc.code != _OVERLOADED_ERROR_CODE or attempt >= _OVERLOADED_RETRIES:
                    raise
                await self._sleep_before_overload_retry(attempt)
        raise CodexAppServerProtocolError("unreachable overloaded retry state")

    async def _sleep_before_overload_retry(self, attempt: int) -> None:
        delay = min(0.25 * (2**attempt), self.timeout)
        jitter = random.uniform(0.0, min(0.1, delay))
        await asyncio.sleep(delay + jitter)

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise CodexAppServerUnavailableError("app-server transport is not connected")
        try:
            await asyncio.wait_for(
                self._ws.send(json.dumps(payload)),
                timeout=self.timeout,
            )
        except (OSError, TimeoutError, WebSocketException) as exc:
            raise CodexAppServerUnavailableError(str(exc)) from exc

    async def _recv_response(self, request_id: int) -> Any:
        if self._ws is None:
            raise CodexAppServerUnavailableError("app-server transport is not connected")

        while True:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout)
            except (OSError, TimeoutError, WebSocketException) as exc:
                raise CodexAppServerUnavailableError(str(exc)) from exc
            try:
                message = json.loads(raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise CodexAppServerProtocolError("invalid JSON-RPC message") from exc

            if message.get("id") != request_id:
                continue

            error = message.get("error")
            if isinstance(error, dict):
                raise CodexAppServerRequestError(
                    code=error.get("code") if isinstance(error.get("code"), int) else None,
                    message=str(error.get("message") or "Codex app-server request failed"),
                    data=error.get("data"),
                )
            if "result" not in message:
                raise CodexAppServerProtocolError("JSON-RPC response missing result")
            return message["result"]

    async def list_threads(self, *, limit: int = 100) -> list[dict[str, Any]]:
        result = await self._request(
            "thread/list",
            {
                "limit": limit,
                "sortKey": "updated_at",
                "sortDirection": "desc",
                "archived": False,
            },
        )
        if not isinstance(result, dict):
            raise CodexAppServerProtocolError("thread/list returned a non-object")
        data = result.get("data", [])
        if not isinstance(data, list):
            raise CodexAppServerProtocolError("thread/list returned invalid data")
        return [item for item in data if isinstance(item, dict)]

    async def read_thread(self, thread_id: str) -> dict[str, Any]:
        result = await self._request(
            "thread/read", {"threadId": thread_id, "includeTurns": False}
        )
        thread = result.get("thread") if isinstance(result, dict) else None
        if not isinstance(thread, dict):
            raise CodexAppServerProtocolError("thread/read returned no thread")
        return thread

    async def resume_thread(self, thread_id: str) -> dict[str, Any]:
        result = await self._request(
            "thread/resume",
            {
                "threadId": thread_id,
                "persistExtendedHistory": True,
                "excludeTurns": True,
            },
        )
        thread = result.get("thread") if isinstance(result, dict) else None
        if not isinstance(thread, dict):
            raise CodexAppServerProtocolError("thread/resume returned no thread")
        return thread

    async def set_thread_name(self, thread_id: str, name: str) -> None:
        result = await self._request(
            "thread/name/set",
            {
                "threadId": thread_id,
                "name": name,
            },
        )
        if not isinstance(result, dict):
            raise CodexAppServerProtocolError("thread/name/set returned a non-object")

    async def start_turn(
        self,
        thread_id: str,
        text: str,
        *,
        extra_input: list[dict[str, Any]] | None = None,
    ) -> CodexTurnSubmission:
        input_items = [
            {"type": "text", "text": text, "text_elements": []},
            *(extra_input or []),
        ]
        try:
            result = await self._request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": input_items,
                },
            )
        except CodexAppServerRequestError as exc:
            if _looks_busy(exc.message):
                raise CodexAppServerBusyError(exc.message) from exc
            raise

        turn = result.get("turn") if isinstance(result, dict) else None
        if not isinstance(turn, dict):
            raise CodexAppServerProtocolError("turn/start returned no turn")
        turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id:
            raise CodexAppServerProtocolError("turn/start returned no turn id")
        return CodexTurnSubmission(thread_id=thread_id, turn_id=turn_id)

    async def submit_turn(
        self,
        thread_id: str,
        text: str,
        *,
        extra_input: list[dict[str, Any]] | None = None,
    ) -> CodexTurnSubmission:
        """Ensure a thread is idle, then submit a user turn."""
        listed_thread = await self._find_listed_thread(thread_id)
        thread = listed_thread if listed_thread is not None else await self.read_thread(thread_id)

        status_type = _status_type(thread)
        if status_type == "active":
            raise CodexAppServerBusyError("Codex app-server thread is active")
        if status_type == "notLoaded":
            thread = await self.resume_thread(thread_id)
            if _status_type(thread) == "active":
                raise CodexAppServerBusyError("Codex app-server thread is active")

        return await self.start_turn(thread_id, text, extra_input=extra_input)

    async def _find_listed_thread(self, thread_id: str) -> dict[str, Any] | None:
        for thread in await self.list_threads():
            if thread.get("id") == thread_id:
                return thread
        return None


async def submit_turn_to_app_server(
    url: str,
    thread_id: str,
    text: str,
    *,
    timeout: float = 3.0,
    extra_input: list[dict[str, Any]] | None = None,
) -> CodexTurnSubmission:
    """Submit text to a Codex app-server thread."""
    async with CodexAppServerClient(url, timeout=timeout) as client:
        return await client.submit_turn(thread_id, text, extra_input=extra_input)


async def set_thread_name_on_app_server(
    url: str,
    thread_id: str,
    name: str,
    *,
    timeout: float = 3.0,
) -> None:
    """Set a Codex app-server thread's user-facing name."""
    async with CodexAppServerClient(url, timeout=timeout) as client:
        await client.set_thread_name(thread_id, name)


async def read_thread_names_from_app_server(
    url: str,
    thread_ids: set[str],
    *,
    timeout: float = 3.0,
) -> dict[str, str]:
    """Read user-facing thread names for known Codex app-server threads."""
    if not thread_ids:
        return {}

    async with CodexAppServerClient(url, timeout=timeout) as client:
        threads = await client.list_threads(limit=max(100, len(thread_ids)))
        names = _names_from_threads(threads, thread_ids)

        for thread_id in sorted(thread_ids - set(names)):
            try:
                thread = await client.read_thread(thread_id)
            except CodexAppServerError:
                continue
            name = _thread_name(thread)
            if name:
                names[thread_id] = name
        return names


def _names_from_threads(
    threads: list[dict[str, Any]],
    thread_ids: set[str],
) -> dict[str, str]:
    names: dict[str, str] = {}
    for thread in threads:
        thread_id = thread.get("id")
        if not isinstance(thread_id, str) or thread_id not in thread_ids:
            continue
        name = _thread_name(thread)
        if name:
            names[thread_id] = name
    return names


def _thread_name(thread: dict[str, Any]) -> str:
    name = thread.get("name")
    return name.strip() if isinstance(name, str) else ""


def _status_type(thread: dict[str, Any]) -> str:
    status = thread.get("status")
    if isinstance(status, dict):
        status_type = status.get("type")
        return status_type if isinstance(status_type, str) else ""
    if isinstance(status, str):
        return status
    return ""


def _unix_socket_path_from_url(url: str) -> str:
    raw_path = url.removeprefix("unix://")
    if not raw_path:
        codex_home = Path(os.getenv("CODEX_HOME") or Path.home() / ".codex")
        return str(codex_home / _DEFAULT_UNIX_SOCKET_RELATIVE)
    return str(Path(raw_path).expanduser())


def _looks_busy(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "active",
            "busy",
            "already running",
            "cannot accept",
            "same-turn",
        )
    )
