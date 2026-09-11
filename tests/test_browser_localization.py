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
