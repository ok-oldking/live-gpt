"""Extract narration and portable rich HTML from the same reply DOM."""

import re
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit


def _web_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


class _ReplyLinks(HTMLParser):
    """Keep existing markup and make bare web URLs clickable outside code."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.urls: set[str] = set()
        self._literal_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(self.get_starttag_text())
        if tag in ("a", "code", "pre"):
            self._literal_depth += 1
        if tag == "a":
            self.urls.add(dict(attrs).get("href") or "")

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f"</{tag}>")
        if tag in ("a", "code", "pre"):
            self._literal_depth = max(0, self._literal_depth - 1)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(self.get_starttag_text())

    def handle_data(self, data: str) -> None:
        if self._literal_depth:
            self.parts.append(escape(data))
            return
        offset = 0
        for match in re.finditer(r'https?://[^\s<>"\u3000]+', data):
            url = match.group().rstrip(".,;:!?)\"'。，；：！？）")
            if not _web_url(url):
                continue
            self.parts.append(escape(data[offset:match.start()]))
            self.parts.append(f'<a href="{escape(url, quote=True)}">{escape(url)}</a>')
            self.urls.add(url)
            offset = match.start() + len(url)
        self.parts.append(escape(data[offset:]))


def reply_html_with_links(reply_html: str, text: str, links: object, sources_label: str) -> str:
    """Keep collected citation URLs visible even when they live outside the body."""
    parser = _ReplyLinks()
    parser.feed(reply_html or "<p>" + escape(text).replace("\n", "<br>") + "</p>")
    parser.close()
    sources: list[str] = []
    if isinstance(links, (list, tuple)):
        for link in links:
            if not isinstance(link, (list, tuple)) or len(link) != 2:
                continue
            label, url = link
            if not isinstance(url, str) or not _web_url(url) or url in parser.urls:
                continue
            parser.urls.add(url)
            sources.append(f'<a href="{escape(url, quote=True)}">{escape(str(label or url))}</a>')
    if sources:
        parser.parts.append(f'<p><strong>{escape(sources_label)}</strong><br>' + " · ".join(sources) + "</p>")
    return "".join(parser.parts)


class _CodeBlocks(HTMLParser):
    """Replace preformatted blocks with cards while retaining exact copy text."""

    def __init__(self, plain_text_label: str) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.blocks: list[tuple[str, str]] = []
        self._code: list[str] | None = None
        self._language = plain_text_label
        self._plain_text_label = plain_text_label

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre" and self._code is None:
            self._code = []
            self._language = dict(attrs).get("data-live-gpt-language") or self._plain_text_label
        elif self._code is not None:
            if tag == "br":
                self._code.append("\n")
        else:
            self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag: str) -> None:
        if self._code is not None:
            if tag == "pre":
                code = "".join(self._code)
                index = len(self.blocks)
                self.blocks.append((self._language, code))
                self.parts.append(
                    '<table width="100%" border="0" cellspacing="0" cellpadding="0" '
                    'style="margin-top: 24px; margin-bottom: 18px;">'
                    '<tr><td style="padding: 16px 20px 8px 20px;"><b><a style="color: #f5f7ff; text-decoration: none;" name="live-gpt-code-' + str(index) + '">'
                    + escape(self._language) + '</a></b></td><td width="60" style="padding: 16px 20px 8px 0;">&nbsp;</td></tr>'
                    '<tr><td colspan="2" style="padding: 8px 20px 18px 20px;"><pre style="margin: 0; white-space: pre;">'
                    + escape(code) + '</pre></td></tr></table><br>'
                )
                self._code = None
        else:
            self.parts.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._code is not None:
            if tag == "br":
                self._code.append("\n")
        else:
            self.parts.append(self.get_starttag_text())

    def handle_data(self, data: str) -> None:
        if self._code is not None:
            self._code.append(data)
        else:
            self.parts.append(escape(data))


def reply_code_blocks(reply_html: str, plain_text_label: str) -> tuple[str, list[tuple[str, str]]]:
    parser = _CodeBlocks(plain_text_label)
    parser.feed(reply_html)
    parser.close()
    return "".join(parser.parts), parser.blocks


RESPONSE_CONTENT_SCRIPT = r"""
element => {
    const citationSelector = [
        '[data-testid="webpage-citation-pill"]',
        '[data-testid="webpage-citation-card"]',
        'span[data-search-result-target]:has([data-testid="chatgpt-citation"])',
        '[data-testid="chatgpt-citation"]', '[data-content-reference-start]',
        '[data-d-component="popover-trigger"]:has([data-d-component="badge"])',
        '[data-d-component="badge"][data-d-hoverable]'
    ].join(',');
    const webURL = value => {
        if (typeof value !== 'string' || !value.trim()) return null;
        try {
            const url = new URL(value, element.ownerDocument.baseURI);
            return ['http:', 'https:'].includes(url.protocol) ? url.href : null;
        } catch (_) { return null; }
    };
    const citations = new Map();
    for (const node of element.querySelectorAll(citationSelector)) {
        if (node.parentElement && node.parentElement.closest(citationSelector)) continue;
        const label = (node.textContent || '').replace(/\s+/g, ' ').trim() || 'Source';
        const links = new Map();
        for (const anchor of [node, node.closest('a[href]'), ...node.querySelectorAll('a[href], [data-url], [data-href]')].filter(Boolean)) {
            const url = webURL(anchor.getAttribute('href') || anchor.getAttribute('data-url') || anchor.getAttribute('data-href'));
            if (url) links.set(url, (anchor.textContent || label).trim());
        }
        // Citation popovers can keep their source URLs in component props
        // until opened. Read only the props near this citation, without clicks.
        if (!links.size) {
            const fiberKey = Object.keys(node).find(key => key.startsWith('__reactFiber$'));
            let fiber = fiberKey && node[fiberKey];
            for (let level = 0; fiber && level < 8; level++, fiber = fiber.return) {
                const seen = new Set();
                let budget = 250;
                const visit = (value, depth) => {
                    if (!value || typeof value !== 'object' || depth > 6 || budget-- <= 0
                        || seen.has(value) || value instanceof Node) return;
                    seen.add(value);
                    for (const [key, child] of Object.entries(value)) {
                        if (['url', 'href', 'source_url'].includes(key)) {
                            const url = webURL(child);
                            if (url) links.set(url, String(value.title || value.name || label));
                        } else if (!['ref', '_owner', 'return', 'stateNode'].includes(key)) visit(child, depth + 1);
                    }
                };
                visit(fiber.memoizedProps, 0);
                if (links.size || fiber.stateNode === element) break;
            }
        }
        if (!links.size) {
            const chat = new URL(element.ownerDocument.URL);
            // Some closed popovers expose no source URL. Make the badge open
            // the original conversation, where its sources can be inspected.
            if (chat.protocol === 'https:' && ['chatgpt.com', 'www.chatgpt.com'].includes(chat.hostname)
                && chat.pathname.startsWith('/c/')) links.set(chat.href, `${label} (ChatGPT)`);
        }
        for (const [url, title] of links) if (!citations.has(url)) citations.set(url, title);
    }
    const clone = element.cloneNode(true);
    for (const block of clone.querySelectorAll('[data-markdown-copy="code-block"], pre')) {
        const header = block.querySelector('[data-markdown-copy="exclude"]');
        const label = header && (header.querySelector('.truncate') || Array.from(header.children).find(
            child => child.tagName === 'DIV' && !child.querySelector('button')
        ));
        const code = block.querySelector('code');
        const language = label && label.textContent.trim() ||
            (code && code.className.match(/(?:^|\s)language-([^\s]+)/) || [])[1];
        if (language) block.setAttribute('data-live-gpt-language', language);
    }
    clone.querySelectorAll([
        '[data-markdown-copy="exclude"]',
        '[data-d-component="shimmer-text"]',
        '[class*="cadencedShimmer-"]',
        '[aria-hidden="true"]', '[hidden]',
        '[data-testid="webpage-citation-pill"]',
        '[data-testid="webpage-citation-card"]',
        'span[data-search-result-target]:has([data-testid="chatgpt-citation"])',
        '[data-testid="chatgpt-citation"]',
        '[data-content-reference-start]',
        '[data-d-component="popover-trigger"]:has([data-d-component="badge"])',
        '[data-d-component="badge"][data-d-hoverable]',
        'button', '[role="button"]:not(a[href])',
        'script', 'style', 'iframe', 'svg', '.sr-only'
    ].join(',')).forEach(node => node.remove());

    const escape = text => text.replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const allowed = new Set(['p', 'br', 'strong', 'b', 'em', 'i', 'u', 's',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote',
        'pre', 'code', 'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
        'hr', 'sub', 'sup']);
    const blocks = new Set(['p', 'div', 'section', 'article', 'li', 'blockquote',
        'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr']);
    const serialize = node => {
        if (node.nodeType === Node.TEXT_NODE) {
            return {text: node.textContent, html: escape(node.textContent)};
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return {text: '', html: ''};
        const tag = node.tagName.toLowerCase();
        if (tag === 'pre' || node.getAttribute('data-markdown-copy') === 'code-block') {
            const code = node.querySelector('code') || node;
            const text = code.textContent || '';
            const language = node.getAttribute('data-live-gpt-language');
            const attrs = language ? ` data-live-gpt-language="${escape(language)}"` : '';
            return {text: text + '\n', html: `<pre${attrs}><code>${escape(text)}</code></pre>`};
        }
        const parts = Array.from(node.childNodes, serialize);
        let text = parts.map(part => part.text).join('');
        let html = parts.map(part => part.html).join('');
        if (tag === 'p' && !text.trim() && !html.includes('<br>')) return {text: '', html: ''};
        if (tag === 'br' || tag === 'hr') text = '\n';
        else if (tag === 'td' || tag === 'th') text += '\t';
        else if (blocks.has(tag) && !node.hasAttribute('data-d-inline')) text += '\n';
        if (tag === 'a' || node.matches('[data-d-component="link"]')) {
            let url;
            try { url = new URL(node.getAttribute('href') || node.getAttribute('data-href') || node.getAttribute('data-url'), element.ownerDocument.baseURI); }
            catch (_) {}
            if (url && ['http:', 'https:'].includes(url.protocol)) {
                html = `<a href="${escape(url.href)}" target="_blank" rel="noopener noreferrer">${html}</a>`;
            }
        } else if (node.hasAttribute('data-d-default-strong')) {
            html = `<strong>${html}</strong>`;
        } else if (node.hasAttribute('data-d-default-emphasis')) {
            html = `<em>${html}</em>`;
        } else if (allowed.has(tag)) {
            let attrs = '';
            if (tag === 'table') attrs = ' border="1" cellspacing="0" cellpadding="6"';
            if (tag === 'td' || tag === 'th') {
                for (const attr of ['colspan', 'rowspan']) {
                    const value = node.getAttribute(attr);
                    if (/^[1-9][0-9]*$/.test(value || '')) attrs += ` ${attr}="${value}"`;
                }
            }
            if (tag === 'ol' && /^\d+$/.test(node.getAttribute('start') || '')) {
                attrs = ` start="${node.getAttribute('start')}"`;
            }
            html = ['br', 'hr'].includes(tag) ? `<${tag}>` : `<${tag}${attrs}>${html}</${tag}>`;
        }
        return {text, html};
    };
    const result = serialize(clone);
    result.text = result.text.replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
    if (citations.size) {
        result.html += '<p>' + Array.from(citations, ([url, label]) =>
            `<a href="${escape(url)}" target="_blank" rel="noopener noreferrer">${escape(label)}</a>`
        ).join(' · ') + '</p>';
    }
    return result;
}
"""
