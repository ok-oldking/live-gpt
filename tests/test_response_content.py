import unittest

from live_gpt.response_content import reply_code_blocks, reply_html_with_links


class ReplyLinksTests(unittest.TestCase):
    def test_sources_are_visible_once_and_escaped(self):
        result = reply_html_with_links('<p><a href="https://example.com/docs">Docs</a></p>', "", (
            ("Docs", "https://example.com/docs"),
            ("Forever +1", "https://example.com/quest?x=1&y=2"),
            ("Duplicate", "https://example.com/quest?x=1&y=2"),
            ('<script>bad</script>', 'javascript:alert(1)'),
        ), "Sources")
        self.assertEqual(result.count('href="https://example.com/docs"'), 1)
        self.assertEqual(result.count('href="https://example.com/quest?x=1&amp;y=2"'), 1)
        self.assertIn("Forever +1</a>", result)
        self.assertNotIn("javascript:", result)
        self.assertNotIn("<script>", result)

    def test_bare_urls_are_clickable_but_code_is_literal(self):
        result = reply_html_with_links('<p>Visit https://example.com/docs.</p>'
                                      '<pre><code>https://example.com/code</code></pre>', "", (), "Sources")
        self.assertIn('<a href="https://example.com/docs">https://example.com/docs</a>.', result)
        self.assertNotIn('href="https://example.com/code"', result)

    def test_plain_text_fallback_still_displays_source_links(self):
        result = reply_html_with_links("", "Reply\nNext paragraph", (("Source", "https://example.com/quest"),), "来源")
        self.assertIn("Reply<br>Next paragraph", result)
        self.assertIn('<a href="https://example.com/quest">Source</a>', result)
        self.assertIn("来源", result)


class ReplyCodeBlockTests(unittest.TestCase):
    def test_blocks_keep_exact_whitespace_unicode_and_decoded_entities(self):
        html, blocks = reply_code_blocks('<p>Before</p><pre data-live-gpt-language="Python">'
                                        '<code>if x &lt; 2:\n\tprint(&quot;中文 &amp; text&quot;)\n</code></pre>'
                                        '<p>After</p><code>inline</code>', "Plain text")
        self.assertEqual(blocks, [("Python", 'if x < 2:\n\tprint("中文 & text")\n')])
        self.assertIn('name="live-gpt-code-0"', html)
        self.assertIn("<p>Before</p>", html)
        self.assertIn("<p>After</p>", html)
        self.assertIn("<code>inline</code>", html)

    def test_multiple_blocks_have_separate_copy_targets(self):
        html, blocks = reply_code_blocks('<pre><code>first\nline</code></pre>'
                                        '<pre data-live-gpt-language="SQL"><code>second</code></pre>', "Plain text")
        self.assertEqual(blocks, [("Plain text", "first\nline"), ("SQL", "second")])
        self.assertIn('name="live-gpt-code-0"', html)
        self.assertIn('name="live-gpt-code-1"', html)
