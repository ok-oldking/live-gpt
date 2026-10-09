"""Expose ChatGPT CDP targets without initializing unrelated browser tabs."""
from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from typing import Any

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .browser_discovery import is_chatgpt_url
from .logger import Logger

logger = Logger.get_logger(__name__)


class _TargetFilter:
    def __init__(self) -> None:
        self._next_id = -1
        self._internal: dict[int, str | None] = {}
        self._ignored_sessions: set[str] = set()
        self._attached: dict[str, str] = {}
        self._attaching: set[str] = set()
        self._known_targets: dict[str, dict] = {}
        self._auto_attach_requests: dict[int, bool] = {}
        self._page_attachments_enabled = False

    def send(self, message: dict) -> dict:
        if message.get("method") == "Target.setAutoAttach" and not message.get("sessionId"):
            params = dict(message.get("params", {}))
            self._auto_attach_requests[message["id"]] = params.get("autoAttach", False)
            # Page attachment has one owner: this relay. Native auto-attach
            # racing Target.attachToTarget creates two sessions for one page
            # and crashes Playwright with its "Duplicate target" assertion.
            native_filter = params.get("filter", [
                {"type": "browser", "exclude": True},
                {"type": "tab", "exclude": True},
                {},
            ])
            params["filter"] = [{"type": "page", "exclude": True}, *native_filter]
            return {**message, "params": params}
        return message

    def _attach_page(self, target: dict) -> list[dict]:
        target_id = target["targetId"]
        if (self._page_attachments_enabled and target.get("type") == "page"
                and is_chatgpt_url(target.get("url", ""))
                and target_id not in self._attached and target_id not in self._attaching):
            return [self.command("Target.attachToTarget", {"targetId": target_id, "flatten": True})]
        return []

    def command(self, method: str, params: dict, **kwargs: Any) -> dict:
        request_id = self._next_id
        self._next_id -= 1
        target_id = params.get("targetId") if method == "Target.attachToTarget" else None
        self._internal[request_id] = target_id
        if target_id:
            self._attaching.add(target_id)
        return {"id": request_id, "method": method, "params": params, **kwargs}

    def receive(self, message: dict) -> tuple[bool, list[dict]]:
        request_id = message.get("id")
        if request_id in self._internal:
            target_id = self._internal.pop(request_id)
            if target_id and "error" in message:
                self._attaching.discard(target_id)
            # Successful replies may precede attachedToTarget. Keep the target
            # reserved until that event arrives, even if its title changes.
            return False, []
        if request_id in self._auto_attach_requests and not message.get("sessionId"):
            enabled = self._auto_attach_requests.pop(request_id)
            if "error" not in message:
                self._page_attachments_enabled = enabled
                commands = [command for target in self._known_targets.values()
                            for command in self._attach_page(target)]
                return True, commands
        if message.get("sessionId") in self._ignored_sessions:
            return False, []
        method, params = message.get("method"), message.get("params", {})
        if method == "Target.attachedToTarget" and not message.get("sessionId"):
            target, session_id = params["targetInfo"], params["sessionId"]
            target_id = target["targetId"]
            if target.get("type") == "page" and target_id in self._attached:
                # Be defensive about a second session, but do not detach the
                # original session or forward a close for its live page.
                if self._attached[target_id] == session_id:
                    return False, []
                self._ignored_sessions.add(session_id)
                return False, [self.command("Target.detachFromTarget", {"sessionId": session_id})]
            if target.get("type") == "page" and not is_chatgpt_url(target.get("url", "")):
                self._ignored_sessions.add(session_id)
                # Auto-attach pauses new pages. Resume them before detaching;
                # never navigate, reload, or close the user's unrelated tabs.
                commands = []
                if params.get("waitingForDebugger"):
                    commands.append(self.command(
                        "Runtime.runIfWaitingForDebugger", {}, sessionId=session_id,
                    ))
                commands.append(self.command("Target.detachFromTarget", {"sessionId": session_id}))
                return False, commands
            if target.get("type") == "page":
                self._attaching.discard(target_id)
                self._attached[target_id] = session_id
        elif method == "Target.detachedFromTarget" and not message.get("sessionId"):
            session_id = params.get("sessionId")
            if session_id in self._ignored_sessions:
                self._ignored_sessions.discard(session_id)
                return False, []
            self._attached = {key: value for key, value in self._attached.items() if value != session_id}
        elif method in {"Target.targetCreated", "Target.targetInfoChanged"} and not message.get("sessionId"):
            target = params["targetInfo"]
            target_id = target["targetId"]
            self._known_targets[target_id] = target
            return True, self._attach_page(target)
        elif method == "Target.targetDestroyed" and not message.get("sessionId"):
            target_id = params["targetId"]
            self._known_targets.pop(target_id, None)
            self._attaching.discard(target_id)
        return True, []


class ChatGPTCDPBridge:
    """Own a loopback-only relay for one Playwright browser connection."""

    def __init__(self, endpoint: str) -> None:
        self.upstream_endpoint = endpoint
        self.endpoint = ""
        self.connected = threading.Event()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="chatgpt-cdp-bridge", daemon=True)

    def start(self) -> str:
        self._thread.start()
        if not self._ready.wait(5):
            self.close()
            raise TimeoutError("Could not start browser connection relay")
        if self._error is not None:
            raise self._error
        return self.endpoint

    def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as error:
            self._error = error
            logger.error("Browser connection relay failed", error)
        finally:
            self._ready.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handle, "127.0.0.1", 0, origins=[None], max_size=None, close_timeout=1) as server:
            self.endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            self._ready.set()
            await self._stop.wait()

    async def _handle(self, downstream: Any) -> None:
        relay = asyncio.create_task(self._relay(downstream))
        stopping = asyncio.create_task(self._stop.wait())
        disconnected = asyncio.create_task(downstream.wait_closed())
        try:
            await asyncio.wait((relay, stopping, disconnected), return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (relay, stopping, disconnected):
                task.cancel()
            results = await asyncio.gather(relay, stopping, disconnected, return_exceptions=True)
            if isinstance(results[0], Exception) and not isinstance(results[0], ConnectionClosed):
                logger.warning(f"Browser connection relay disconnected: {results[0]}")
            await downstream.close()

    async def _relay(self, downstream: Any) -> None:
        endpoint = self.upstream_endpoint
        if endpoint.startswith(("http://", "https://")):
            endpoint = await asyncio.to_thread(self._resolve_endpoint, endpoint)
        # Approval has no fixed deadline here. The monitor bounds the handshake
        # and cancels this relay on explicit retry or exit.
        async with connect(endpoint, proxy=None, open_timeout=None, max_size=None, close_timeout=1) as upstream:
            self.connected.set()
            logger.info("Browser approved the remote debugging connection")
            targets = _TargetFilter()
            await upstream.send(json.dumps(targets.command("Target.setDiscoverTargets", {"discover": True})))

            async def forward_requests() -> None:
                async for message in downstream:
                    await upstream.send(json.dumps(targets.send(json.loads(message))))

            async def forward_events() -> None:
                async for raw in upstream:
                    forward, commands = targets.receive(json.loads(raw))
                    for command in commands:
                        await upstream.send(json.dumps(command))
                    if forward:
                        await downstream.send(raw)

            tasks = [asyncio.create_task(forward_requests()), asyncio.create_task(forward_events())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _resolve_endpoint(endpoint: str) -> str:
        with urllib.request.urlopen(f"{endpoint.rstrip('/')}/json/version", timeout=3) as response:
            return json.load(response)["webSocketDebuggerUrl"]
