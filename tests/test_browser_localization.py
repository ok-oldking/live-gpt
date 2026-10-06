"""DOM regressions; run with LIVE_GPT_TEST_BROWSER=msedge (or chrome)."""
from __future__ import annotations

import os
import unittest

from playwright.sync_api import sync_playwright

from live_gpt.browser import BrowserMonitor


@unittest.skipUnless(os.environ.get("LIVE_GPT_TEST_BROWSER"), "Browser channel not selected")
class BrowserLocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            channel=os.environ["LIVE_GPT_TEST_BROWSER"], headless=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.page = self.browser.new_page()
        self.addCleanup(self.page.close)

    def test_work_usage_banner_keeps_reset_details_and_omits_action_buttons(self):
        title = "You’ve reached your 5-hour Work usage limit"
        description = "Upgrade to Pro to remove the 5-hour limit, add credits to continue now, or wait for your usage to reset at 2:53 PM."
        self.page.set_content(f'''
            <form data-thread-find-composer="true">
                <aside role="status" aria-live="polite">
                    <div><div><h3>{title}</h3><div>{description}</div></div>
                    <div><button>Add credits</button><button>Upgrade</button></div></div>
                </aside>
                <div data-composer-markdown contenteditable="true" role="textbox">Unsent prompt</div>
            </form>
        ''')
        message = BrowserMonitor._usage_limit_message(self.page)
        self.assertEqual(message, f"{title} {description}")
        snapshot = BrowserMonitor._response_snapshot(self.page, None)
        self.assertEqual(snapshot.error_message, message)
        self.assertFalse(snapshot.is_generating)
        with self.assertRaisesRegex(RuntimeError, "5-hour Work usage limit"):
            BrowserMonitor._send_to_chatgpt_page(self.page, "New prompt")
        self.assertEqual(BrowserMonitor._read_composer_text(self.page), "Unsent prompt")

    def test_hidden_limits_and_assistant_text_do_not_block_send(self):
        self.page.set_content('''
            <form data-thread-find-composer="true">
                <aside hidden role="status">You’ve reached your usage limit</aside>
                <aside role="status">You have messages remaining</aside>
            </form>
            <div data-turn-key="reply">
                <div data-markdown-text-style="assistant-message">You’ve reached your usage limit</div>
            </div>
        ''')
        self.assertEqual(BrowserMonitor._usage_limit_message(self.page), "")
        self.assertEqual(BrowserMonitor._response_snapshot(self.page, None).error_message, "")

    def test_chinese_usage_limit_alert_is_reported(self):
        message = "你已达到 Work 使用限额，请等待额度重置。"
        self.page.set_content(f'<div role="alert">{message}</div>')
        self.assertEqual(BrowserMonitor._usage_limit_message(self.page), message)

    def test_send_uses_active_chinese_composer_and_waits_until_enabled(self):
        for attribute in ('data-testid="send-button"', 'id="composer-submit-button"'):
            with self.subTest(attribute=attribute):
                self.page.set_content(f'''
                    <form hidden onsubmit="window.wrongSent=true; return false">
                        <div id="prompt-textarea" contenteditable="true"></div>
                        <button data-testid="send-button" type="submit">发送</button>
                    </form>
                    <form onsubmit="window.wrongSent=true; return false">
                        <button data-testid="send-button" type="submit">Other form</button>
                    </form>
                    <div role="dialog">
                      <form onsubmit="window.sentText=document.querySelector('#active').innerText;
                                      window.sendCount++; return false">
                        <div data-composer-surface="true">
                          <div id="active" contenteditable="true" role="textbox" oninput="
                            setTimeout(() => document.querySelector('#real-send').removeAttribute('aria-disabled'), 100)"></div>
                        </div>
                        <div hidden><button type="submit" data-testid="send-button">Hidden copy</button></div>
                        <span id="real-send" aria-disabled="true">
                          <button {attribute} type="submit" aria-label="发送提示">发送</button>
                        </span>
                        <button type="button" aria-label="启动语音功能"
                                onclick="window.wrongSent=true">语音</button>
                      </form>
                    </div>
                ''')
                self.page.evaluate("window.wrongSent=false; window.sendCount=0; window.sentText=''")
                BrowserMonitor._send_to_chatgpt_page(self.page, "请读这条消息", preserve_attachments=True)
                self.assertEqual(self.page.evaluate("window.sentText"), "请读这条消息")
                self.assertEqual(self.page.evaluate("window.sendCount"), 1)
                self.assertFalse(self.page.evaluate("window.wrongSent"))

    def test_markdown_composer_sends_with_unlabelled_submit_id(self):
        self.page.set_content('''
            <form data-thread-find-composer="true" hidden>
                <div data-composer-markdown contenteditable="true" role="textbox"></div>
                <button type="submit">Hidden send</button>
            </form>
            <form onsubmit="window.wrongSent=true; return false">
                <div contenteditable="true" role="textbox">Other editor</div>
                <button type="submit">Other send</button>
            </form>
            <form data-thread-find-composer="true" data-composer-placement="home"
                  onsubmit="window.sendCount++; return false">
                <div data-composer-markdown contenteditable="true" role="textbox"
                     aria-multiline="true" aria-label="使用 ChatGPT Work"></div>
                <button type="button" onclick="window.wrongSent=true">听写</button>
                <button type="submit" aria-label="发送">发送</button>
            </form>
        ''')
        self.page.evaluate("window.sendCount=0; window.wrongSent=false")
        BrowserMonitor._send_to_chatgpt_page(self.page, "Send this message", preserve_attachments=True)
        self.assertEqual(self.page.evaluate("window.sendCount"), 1)
        self.assertFalse(self.page.evaluate("window.wrongSent"))
        self.assertEqual(BrowserMonitor._read_composer_text(self.page), "Send this message")
        BrowserMonitor._clear_chatgpt_composer(self.page)
        self.assertEqual(BrowserMonitor._read_composer_text(self.page), "")

    def test_new_turn_layout_completes_and_reads_only_assistant_text(self):
        from unittest.mock import patch
        from live_gpt.browser import _ActiveResponse, _MonitorState
        self.page.set_content("""
            <div data-turn-key="old">
                <div data-chatgpt-search-unit-key="turn-0:assistant">
                    <div data-markdown-text-style="assistant-message">Old reply</div>
                </div>
                <div class="turn-action-controls"><button aria-label="Copy">Copy</button></div>
            </div>
            <div data-turn-key="new">
                <div data-user-message-bubble="true">User question</div>
                <button aria-label="Copy message">Copy message</button>
            </div>
        """)
        before = BrowserMonitor._assistant_turn_marker(self.page)
        self.assertEqual(before, "old")
        self.assertFalse(BrowserMonitor._response_snapshot(self.page, before).has_new_turn)
        self.page.locator('[data-turn-key="new"]').evaluate("""element => {
            element.insertAdjacentHTML('beforeend', `
                <div data-chatgpt-search-unit-key="turn-1:2:assistant">
                    <div data-markdown-text-style="assistant-message">New reply</div>
                </div>
                <div class="turn-action-controls">
                    <button aria-label="Copy">Copy</button>
                    <button aria-label="Read aloud" onclick="window.readClicked=true">Read aloud</button>
                </div>`);
        }""")
        snapshot = BrowserMonitor._response_snapshot(self.page, before)
        self.assertEqual(snapshot.text, "New reply")
        self.assertTrue(snapshot.has_completion_controls)
        self.assertFalse(snapshot.is_generating)
        monitor = BrowserMonitor()
        state = _MonitorState(active_response=_ActiveResponse(
            page=self.page, turn_marker_before=before, started_at=90,
        ))
        # Even with actions present, a generating response must not finish.
        self.page.evaluate("""() => document.body.insertAdjacentHTML('beforeend',
            '<button data-testid="stop-button">Stop</button>')""")
        with patch("live_gpt.browser.time.monotonic", return_value=100) as clock:
            monitor._poll_active_response(state)
            self.assertIsNotNone(state.active_response)
            self.assertIsNone(state.active_response.completion_candidate_at)
            self.page.locator('[data-testid="stop-button"]').evaluate('element => element.remove()')
            monitor._poll_active_response(state)
            clock.return_value = 103
            monitor._poll_active_response(state)
        self.assertIsNone(state.active_response)
        self.assertIsNotNone(state.active_reading)
        self.assertTrue(self.page.evaluate("window.readClicked"))
        self.assertEqual(state.active_reading.full_text, "New reply")

    def test_search_status_updates_before_assistant_reply_exists(self):
        from live_gpt.browser import _ActiveResponse, _MonitorState
        self.page.set_content("""
            <div data-request-input-activity-root>
                <span role="status">Searching the web</span>
            </div>
        """)
        monitor = BrowserMonitor()
        updates = []
        monitor.response_changed.connect(lambda *args: updates.append(args))
        state = _MonitorState(active_response=_ActiveResponse(page=self.page, turn_marker_before=None))
        monitor._poll_active_response(state)
        self.assertEqual(updates, [("Searching the web", "")])
        self.page.locator('[role="status"]').evaluate("element => element.textContent='Reading sources'")
        monitor._poll_active_response(state)
        self.assertEqual(updates[-1], ("Reading sources", ""))
        self.assertIsNotNone(state.active_response)
        self.assertIsNone(state.active_response.completion_candidate_at)

    def test_thinking_shimmer_before_new_assistant_turn(self):
        self.page.set_content("""
            <div data-turn-key="old">
              <div data-chatgpt-search-unit-key="old:assistant">
                <div data-markdown-text-style="assistant-message">Old reply</div>
              </div>
            </div>
            <div data-turn-key="new">
              <div data-user-message-bubble>New question</div>
              <span data-chatgpt-agent-turn-start></span>
              <span data-d-component="shimmer-text">Thinking</span>
            </div>
            <form data-thread-find-composer="true"><button aria-label="停止">Stop</button></form>
        """)
        snapshot = BrowserMonitor._response_snapshot(self.page, "old")
        self.assertEqual(snapshot.status, "Thinking")
        self.assertTrue(snapshot.is_generating)
        self.assertEqual(snapshot.text, "")
        self.page.locator('[data-d-component="shimmer-text"]').evaluate(
            "element => element.textContent='Searching the web'")
        self.assertEqual(BrowserMonitor._response_snapshot(self.page, "old").status, "Searching the web")

    def test_cadenced_shimmer_status_ignores_sweep_and_streams_updates(self):
        from live_gpt.browser import _ActiveResponse, _MonitorState
        self.page.set_content('''
            <div data-turn-key="old">
              <span class="cadencedShimmer-old cadencedShimmerActive-old">Old activity</span>
            </div>
            <div data-turn-key="new">
              <span id="activity" class="relative inline-block align-top cadencedShimmer-uMTG1d
                  text-size-chat select-none truncate cadencedShimmerActive-DpK60y">正在搜索 6 个网站<span
                  aria-hidden="true" class="cadencedShimmerSweep-ICUAVH"><span
                  class="cadencedShimmerHighlight-VcX29h">正在搜索 6 个网站</span></span></span>
              <span class="cadencedShimmer-other">Inactive activity</span>
              <span hidden class="cadencedShimmer-hidden cadencedShimmerActive-hidden">Hidden activity</span>
            </div>
            <button data-testid="stop-button">Stop</button>
        ''')
        monitor = BrowserMonitor()
        updates = []
        monitor.response_changed.connect(lambda *args: updates.append(args))
        state = _MonitorState(active_response=_ActiveResponse(page=self.page, turn_marker_before="old"))
        monitor._poll_active_response(state)
        self.assertEqual(updates[-1], ("正在搜索 6 个网站", ""))
        self.page.locator('#activity').evaluate("element => element.firstChild.textContent='正在阅读来源'")
        monitor._poll_active_response(state)
        self.assertEqual(updates[-1], ("正在阅读来源", ""))
        self.assertIsNone(state.active_response.completion_candidate_at)
        self.page.locator('#activity').evaluate("element => element.classList.remove('cadencedShimmerActive-DpK60y')")
        self.page.locator('[data-testid="stop-button"]').evaluate("element => element.remove()")
        snapshot = BrowserMonitor._response_snapshot(self.page, "old")
        self.assertFalse(snapshot.is_generating)
        self.assertEqual(snapshot.status, "Waiting for ChatGPT…")

    def test_search_activity_is_not_extracted_as_reply_text(self):
        for reply in ('', '<div class="markdown">Actual reply</div>'):
            with self.subTest(reply=reply):
                self.page.set_content(f'''
                    <section data-testid="conversation-turn-2" data-turn="assistant" data-turn-id="new">
                      <span class="cadencedShimmer-label">已搜索 6 个网站<span aria-hidden="true">已搜索 6 个网站</span></span>
                      {reply}
                    </section>
                ''')
                snapshot = BrowserMonitor._response_snapshot(self.page, "old")
                self.assertEqual(snapshot.text, "Actual reply" if reply else "")

    def test_activity_markdown_and_localized_status_stream_before_final_reply(self):
        from live_gpt.browser import _ActiveResponse, _MonitorState
        self.page.set_content("""
            <div data-turn-key="new">
                <div data-user-message-bubble>Compare these</div>
                <span data-chatgpt-agent-turn-start></span>
                <div data-markdown-animated data-markdown-text-style="assistant-message">I will compare the bonuses.</div>
                <span data-d-component="shimmer-text">检索职业数据</span>
            </div>
        """)
        monitor = BrowserMonitor()
        monitor.set_use_browser_voice(False)
        updates = []
        monitor.response_changed.connect(lambda *args: updates.append(args))
        state = _MonitorState(active_response=_ActiveResponse(page=self.page, turn_marker_before="old"))
        monitor._poll_active_response(state)
        self.assertEqual(updates[-1], ("检索职业数据", "I will compare the bonuses."))
        self.assertIsNotNone(state.active_response)
        self.assertIsNone(state.active_response.completion_candidate_at)
        self.page.locator('[data-d-component="shimmer-text"]').evaluate(
            "element => element.textContent='Comparing results'")
        self.page.locator('[data-markdown-animated]').evaluate(
            "element => element.textContent='I found the current bonuses.'")
        monitor._poll_active_response(state)
        self.assertEqual(updates[-1], ("Comparing results", "I found the current bonuses."))
        self.assertIsNotNone(state.active_response)

    def test_code_block_formatting_is_excluded_from_subtitles_and_local_voice(self):
        from unittest.mock import patch
        from live_gpt.browser import _ActiveResponse, _MonitorState

        for language_label in ("Plain text", "Python", "纯文本"):
            with self.subTest(language_label=language_label):
                self.page.set_content("".join(line.strip() for line in f'''
                    <div data-turn-key="new">
                      <div data-markdown-text-style="assistant-message">
                        <p>把这行放进宏里，点击就能重载界面：</p>
                        <div class="CodeBlock-zu1QM3">
                          <div data-markdown-copy="code-block">
                            <div data-markdown-copy="exclude">
                              <div>{language_label}</div>
                              <button aria-label="Enable word wrap">Wrap</button>
                              <button aria-label="Copy">Copy code</button>
                            </div>
                            <div><code><span>/reload</span></code></div>
                          </div>
                        </div>
                        <p>Plain text 是正文。<code>Python</code> 也是正文。</p>
                      </div>
                      <div class="turn-action-controls"><button aria-label="Copy">Copy</button></div>
                    </div>
                '''.splitlines()))
                snapshot = BrowserMonitor._response_snapshot(self.page, "old")
                expected = "把这行放进宏里，点击就能重载界面：/reloadPlain text 是正文。Python 也是正文。"
                self.assertEqual("".join(snapshot.text.splitlines()).strip(), expected)
                monitor = BrowserMonitor()
                monitor.set_use_browser_voice(False)
                subtitles, speech = [], []
                monitor.response_changed.connect(lambda _status, text: subtitles.append(text))
                monitor.local_voice_updated.connect(lambda text, final: speech.append((text, final)))
                state = _MonitorState(active_response=_ActiveResponse(
                    page=self.page, turn_marker_before="old", started_at=90,
                ))
                with patch("live_gpt.browser.time.monotonic", return_value=100) as clock:
                    monitor._poll_active_response(state)
                    clock.return_value = 103
                    monitor._poll_active_response(state)
                self.assertTrue(subtitles)
                self.assertTrue(all(text == snapshot.text for text in subtitles))
                self.assertEqual(speech, [(snapshot.text, False), (snapshot.text, True)])
                # Extraction must leave the original browser response intact.
                self.assertEqual(
                    self.page.locator('[data-markdown-copy="exclude"]').count(), 1
                )

    def test_completion_and_read_aloud_menu_in_each_language(self):
        for copy, more, read in (
            ("Copy response", "More actions", "Read aloud"),
            ("复制回复", "更多操作", "朗读"),
            ("複製回覆", "更多操作", "朗讀"),
        ):
            with self.subTest(language=read):
                self.page.set_content(f'''
                    <section data-testid="conversation-turn-18"
                             data-turn="assistant" data-turn-id="new">
                      <h4>ChatGPT 说：</h4>
                      <div data-message-author-role="assistant">
                        <div class="markdown"><p>这是回复。</p></div>
                      </div>
                      <button data-testid="copy-turn-action-button"
                              aria-label="{copy}"></button>
                      <button aria-label="{more}" onclick="
                        document.querySelector('[role=menuitem]').hidden=false">
                        ...
                      </button>
                    </section>
                    <button role="menuitem" hidden
                            onclick="window.readClicked=true">{read}</button>
                ''')
                snapshot = BrowserMonitor._response_snapshot(self.page, "old")
                self.assertTrue(snapshot.has_completion_controls)
                self.assertFalse(snapshot.is_generating)
                self.assertEqual(snapshot.text, "这是回复。")
                self.page.evaluate("window.readClicked=false")
                self.assertTrue(BrowserMonitor._click_read_aloud(self.page))
                self.assertTrue(self.page.evaluate("window.readClicked"))

    def test_chinese_generation_and_direct_read_control(self):
        self.page.set_content('''
            <section data-testid="conversation-turn-2" data-turn="assistant"
                     data-turn-id="new">
                <div class="markdown">你好</div>
                <button aria-label="朗读" onclick="window.readClicked=true">播放</button>
            </section>
            <button aria-label="停止生成">停止</button>
        ''')
        self.assertTrue(BrowserMonitor._response_snapshot(self.page, "old").is_generating)
        self.assertTrue(BrowserMonitor._click_read_aloud(self.page))
        self.assertTrue(self.page.evaluate("window.readClicked"))

    def test_read_aloud_nested_label_ignores_hidden_duplicate(self):
        for role, label in (("menuitem", "朗读"), ("menuitemradio", "朗讀"),
                            ("menuitem", "Read aloud")):
            with self.subTest(role=role, label=label):
                self.page.set_content(f'''
                    <section data-testid="conversation-turn-2" data-turn="assistant"
                             data-turn-id="new">
                        <button aria-label="更多操作" onclick="
                            document.querySelector('#actions').hidden=false">...</button>
                    </section>
                    <div id="actions" role="menu" hidden>
                        <div role="menuitem">今天，23:10</div>
                        <div role="menuitem" onclick="window.wrongClicked=true">查看来源</div>
                        <div role="menuitem" onclick="window.wrongClicked=true">新聊天中的分支</div>
                        <div role="{role}" onclick="window.readClicked=true">
                            <svg aria-hidden="true"></svg><div><span>{label}</span></div>
                        </div>
                    </div>
                    <div hidden role="{role}" onclick="window.wrongClicked=true">{label}</div>
                ''')
                self.page.evaluate("window.readClicked=false; window.wrongClicked=false")
                self.assertTrue(BrowserMonitor._click_read_aloud(self.page))
                self.assertTrue(self.page.evaluate("window.readClicked"))
                self.assertFalse(self.page.evaluate("window.wrongClicked"))

    def test_chinese_dictation_start_finish_and_cancel(self):
        self.page.set_content('''
            <form onsubmit="return false">
                <div id="prompt-textarea" contenteditable="true"></div>
                <button type="button" aria-label="启动语音功能"
                        onclick="window.wrongControl=true">语音</button>
                <button type="button" aria-label="开始听写" onclick="
                    document.querySelector('#done').hidden=false;
                    document.querySelector('#cancel').hidden=false">听写</button>
                <button type="button" id="done" aria-label="提交听写" hidden onclick="
                    this.hidden=true;
                    document.querySelector('#cancel').hidden=true;
                    document.querySelector('#prompt-textarea').textContent='你好世界'">完成</button>
                <button type="button" id="cancel" aria-label="取消听写" hidden onclick="
                    this.hidden=true; document.querySelector('#done').hidden=true">取消</button>
            </form>
        ''')
        monitor = BrowserMonitor()
        monitor._start_browser_dictation(self.page)
        self.assertEqual(monitor._finish_browser_dictation(self.page, ""), "你好世界")
        monitor._start_browser_dictation(self.page)
        self.assertEqual(monitor._cancel_browser_dictation(self.page, "原文"), "原文")
        self.assertIsNone(self.page.evaluate("window.wrongControl"))

    def test_chinese_attachment_removal(self):
        self.page.set_content('''
            <button aria-label="移除附件" onclick="this.remove()">移除</button>
            <button aria-label="删除图片" onclick="this.remove()">删除</button>
            <button aria-label="添加文件等">添加</button>
        ''')
        BrowserMonitor._clear_chatgpt_attachments(self.page)
        self.assertEqual(self.page.locator("button").count(), 1)
        self.assertEqual(self.page.locator("button").get_attribute("aria-label"), "添加文件等")
