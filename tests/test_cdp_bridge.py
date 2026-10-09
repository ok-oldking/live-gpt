from __future__ import annotations

import asyncio
import logging
import os
import socket
import tempfile
import time
import unittest

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from websockets.asyncio.server import serve
from websockets.sync.client import connect

from live_gpt.cdp_bridge import ChatGPTCDPBridge, _TargetFilter
from live_gpt.browser import BrowserMonitor
from unittest.mock import patch


class TargetFilterTests(unittest.TestCase):
    @staticmethod
    def enable_page_attachments(targets):
        request = targets.send({"id": 1, "method": "Target.setAutoAttach", "params": {
            "autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True,
        }})
        return request, targets.receive({"id": 1, "result": {}})

    def test_existing_target_change_waits_until_native_page_auto_attach_is_disabled(self):
        targets = _TargetFilter()
        info = {"targetId": "existing", "type": "page", "url": "https://chatgpt.com/c/existing"}
        for method in ("Target.targetCreated", "Target.targetInfoChanged"):
            self.assertEqual(targets.receive({"method": method, "params": {"targetInfo": info}}), (True, []))
        request, (forward, commands) = self.enable_page_attachments(targets)
        self.assertEqual(request["params"]["filter"][0], {"type": "page", "exclude": True})
        self.assertEqual(request["params"]["filter"][1:], [
            {"type": "browser", "exclude": True}, {"type": "tab", "exclude": True}, {},
        ])
        self.assertTrue(forward)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["params"]["targetId"], "existing")

    def test_attachment_reply_before_event_does_not_trigger_a_second_attachment(self):
        targets = _TargetFilter()
        self.enable_page_attachments(targets)
        event = {"method": "Target.targetInfoChanged", "params": {"targetInfo": {
            "targetId": "existing", "type": "page", "url": "https://chatgpt.com/",
        }}}
        _, commands = targets.receive(event)
        targets.receive({"id": commands[0]["id"], "result": {"sessionId": "session"}})
        self.assertEqual(targets.receive(event), (True, []))
        attached = {"method": "Target.attachedToTarget", "params": {
            "targetInfo": event["params"]["targetInfo"], "sessionId": "session",
        }}
        self.assertEqual(targets.receive(attached), (True, []))
        self.assertEqual(targets.receive(event), (True, []))
        self.assertEqual(targets.receive(attached), (False, []))
        attached["params"]["sessionId"] = "duplicate"
        forward, commands = targets.receive(attached)
        self.assertFalse(forward)
        self.assertEqual(commands[0]["method"], "Target.detachFromTarget")
        self.assertEqual(targets.receive({"method": "Target.detachedFromTarget", "params": {
            "targetId": "existing", "sessionId": "duplicate",
        }}), (False, []))
        self.assertEqual(targets._attached["existing"], "session")

    def test_page_scoped_frame_auto_attach_is_not_modified(self):
        targets = _TargetFilter()
        request = {"id": 1, "sessionId": "chat-session", "method": "Target.setAutoAttach", "params": {
            "autoAttach": True, "flatten": True,
        }}
        self.assertEqual(targets.send(request), request)

    def test_unrelated_paused_target_is_resumed_and_detached(self):
        targets = _TargetFilter()
        forward, commands = targets.receive({"method": "Target.attachedToTarget", "params": {
            "targetInfo": {"targetId": "other", "type": "page", "url": "https://example.com"},
            "sessionId": "other-session", "waitingForDebugger": True,
        }})
        self.assertFalse(forward)
        self.assertEqual([c["method"] for c in commands], [
            "Runtime.runIfWaitingForDebugger", "Target.detachFromTarget",
        ])
        self.assertEqual(commands[0]["sessionId"], "other-session")
        for command in commands:
            self.assertEqual(targets.receive({"id": command["id"], "result": {}}), (False, []))
        self.assertEqual(targets.receive({"sessionId": "other-session", "method": "Page.loadEventFired"}), (False, []))
        self.assertEqual(targets.receive({"method": "Target.detachedFromTarget", "params": {
            "sessionId": "other-session",
        }}), (False, []))

    def test_chatgpt_target_and_child_frame_events_are_forwarded(self):
        targets = _TargetFilter()
        self.assertEqual(targets.receive({"method": "Target.attachedToTarget", "params": {
            "targetInfo": {"targetId": "chat", "type": "page", "url": "https://chatgpt.com/c/test"},
            "sessionId": "chat-session", "waitingForDebugger": False,
        }}), (True, []))
        self.assertEqual(targets.receive({"sessionId": "chat-session", "method": "Target.attachedToTarget", "params": {
            "targetInfo": {"targetId": "frame", "type": "iframe", "url": "https://example.com"},
            "sessionId": "frame-session",
        }}), (True, []))

    def test_navigation_to_chatgpt_attaches_once(self):
        targets = _TargetFilter()
        self.enable_page_attachments(targets)
        event = {"method": "Target.targetInfoChanged", "params": {
            "targetInfo": {"targetId": "new", "type": "page", "url": "https://chatgpt.com/"},
        }}
        forward, commands = targets.receive(event)
        self.assertTrue(forward)
        self.assertEqual(commands[0]["method"], "Target.attachToTarget")
        self.assertEqual(targets.receive(event), (True, []))


class BridgeLifecycleTests(unittest.TestCase):
    def test_monitor_retry_restarts_real_playwright_driver_during_pending_approval(self):
        async def scenario():
            connections = asyncio.Queue()
            release = asyncio.Event()
            async def delayed_approval(connection, request):
                await connections.put(connection)
                await release.wait()
            async with serve(lambda connection: connection.wait_closed(), "127.0.0.1", 0,
                             process_request=delayed_approval, close_timeout=1,
                             logger=logging.getLogger("cdp-delayed-approval-test")) as upstream:
                endpoint = f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}"
                monitor = BrowserMonitor()
                try:
                    with patch("live_gpt.browser.discover_cdp_endpoint", return_value=endpoint):
                        monitor.start()
                        await asyncio.wait_for(connections.get(), timeout=5)
                        self.assertTrue(monitor._connection_pending.is_set())
                        monitor.request_retry_connection()
                        await asyncio.wait_for(connections.get(), timeout=5)
                        self.assertTrue(monitor.isRunning())
                        monitor.request_stop()
                        self.assertTrue(await asyncio.to_thread(monitor.wait, 5_000))
                finally:
                    release.set()
                    monitor.request_stop()
                    self.assertTrue(await asyncio.to_thread(monitor.wait, 5_000))
        with self.assertLogs("cdp-delayed-approval-test", level="DEBUG"):
            asyncio.run(scenario())

    def test_close_cancels_upstream_waiting_for_approval(self):
        async def scenario():
            entered = asyncio.Event()
            release = asyncio.Event()
            async def delayed_approval(connection, request):
                entered.set()
                await release.wait()
            async with serve(lambda connection: connection.wait_closed(), "127.0.0.1", 0,
                             process_request=delayed_approval, close_timeout=1,
                             logger=logging.getLogger("cdp-delayed-approval-test")) as upstream:
                bridge = ChatGPTCDPBridge(f"ws://127.0.0.1:{upstream.sockets[0].getsockname()[1]}")
                endpoint = bridge.start()
                downstream = await asyncio.to_thread(connect, endpoint)
                try:
                    await asyncio.wait_for(entered.wait(), timeout=3)
                    self.assertFalse(bridge.connected.is_set())
                    await asyncio.to_thread(bridge.close)
                    self.assertFalse(bridge._thread.is_alive())
                finally:
                    release.set()
                    await asyncio.to_thread(downstream.close)
                    bridge.close()
        # The intentionally aborted handshake is expected to log a close.
        with self.assertLogs("cdp-delayed-approval-test", level="DEBUG"):
            asyncio.run(scenario())


@unittest.skipUnless(os.environ.get("LIVE_GPT_TEST_BROWSER"), "Browser channel not selected")
class BridgeBrowserTests(unittest.TestCase):
    def test_persistent_profile_with_changing_titles_has_one_session_per_page(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as profile, sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                profile, channel=os.environ["LIVE_GPT_TEST_BROWSER"], headless=True,
                args=[f"--remote-debugging-port={port}"],
            )
            try:
                context.route("https://chatgpt.com/**", lambda route: route.fulfill(
                    body="<p>Healthy</p><script>let n=0;setInterval(()=>document.title='Chat '+n++,10)</script>",
                ))
                for index in range(4):
                    context.new_page().goto(f"https://chatgpt.com/c/churn-{index}")
                expected = {f"https://chatgpt.com/c/churn-{index}" for index in range(4)}
                for attempt in range(5):
                    with self.subTest(attempt=attempt):
                        bridge = ChatGPTCDPBridge(f"http://127.0.0.1:{port}")
                        try:
                            connected = playwright.chromium.connect_over_cdp(
                                bridge.start(), no_defaults=True, timeout=5_000,
                            )
                            session = connected.new_browser_cdp_session()
                            for _ in range(10):
                                session.send("Target.getTargets")
                                pages = [page for ctx in connected.contexts for page in ctx.pages]
                                if len(pages) == len(expected):
                                    break
                                time.sleep(0.05)
                            self.assertEqual(len(pages), len(expected))
                            self.assertEqual({page.url for page in pages}, expected)
                            for page in pages:
                                self.assertEqual(page.inner_text("p"), "Healthy")
                            connected.close()
                            self.assertEqual(len(context.pages), 5)
                        finally:
                            bridge.close()
            finally:
                context.close()

    def test_connect_with_unresponsive_unrelated_tab_and_attach_future_chatgpt_tab(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel=os.environ["LIVE_GPT_TEST_BROWSER"], headless=True,
                args=[f"--remote-debugging-port={port}"],
            )
            bridge = ChatGPTCDPBridge(f"http://127.0.0.1:{port}")
            try:
                context = browser.new_context()
                context.route("https://chatgpt.com/**", lambda route: route.fulfill(
                    body="<title>ChatGPT test</title><p>Healthy conversation</p>",
                ))
                healthy = context.new_page()
                healthy.goto("https://chatgpt.com/c/healthy")
                frozen = context.new_page()
                frozen.goto("data:text/html,<title>Busy unrelated tab</title>")
                frozen.evaluate("setTimeout(() => { for (;;) {} }, 100)")
                time.sleep(0.2)
                with self.assertRaises(PlaywrightTimeoutError):
                    playwright.chromium.connect_over_cdp(
                        f"http://127.0.0.1:{port}", no_defaults=True, timeout=1_000,
                    )
                connected = playwright.chromium.connect_over_cdp(
                    bridge.start(), no_defaults=True, timeout=5_000,
                )
                pages = [page for ctx in connected.contexts for page in ctx.pages]
                self.assertEqual([page.url for page in pages], [healthy.url])
                self.assertEqual(pages[0].inner_text("p"), "Healthy conversation")
                new_page = context.new_page()
                new_page.goto("https://chatgpt.com/c/new")
                session = connected.new_browser_cdp_session()
                for _ in range(20):
                    session.send("Target.getTargets")
                    pages = [page for ctx in connected.contexts for page in ctx.pages]
                    if any(page.url == new_page.url for page in pages):
                        break
                    time.sleep(0.05)
                self.assertTrue(any(page.url == new_page.url for page in pages))
                connected.close()
                self.assertTrue(browser.is_connected())
                self.assertEqual(healthy.inner_text("p"), "Healthy conversation")
            finally:
                bridge.close()
                browser.close()
