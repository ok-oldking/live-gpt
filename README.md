# live-gpt
voice and share screen chat with your web chatbot, free of tokens

## Run

```powershell
.\.venv\Scripts\Activate.ps1
live-gpt
```

The app starts as a translucent, always-on-top overlay centered above the
bottom of the screen and also places an icon in the Windows system tray. Drag
any non-button area to reposition it. Use the overlay's **Hide** and **Exit**
buttons, or double-click the tray icon to restore a hidden overlay.

Press and hold the microphone button to start ChatGPT's browser dictation
through Playwright. Releasing it clicks ChatGPT's **Done** control, waits for
the dictated text to appear in ChatGPT's composer, and copies that text into
the Live GPT input. The ChatGPT tab owns microphone capture and speech
recognition, so the browser may ask for microphone permission the first time.
Live GPT does not record audio locally or download a speech-recognition model.

The recognized text remains editable. When the editor contains text, its
paper-plane button sends the current text through Playwright to the selected
ChatGPT tab, and its **X** button clears the editor. The app fills ChatGPT's
composer and clicks its send button, then clears the local text after ChatGPT
accepts it. The **X** button also clears the selected ChatGPT tab's composer.
These browser interactions run without intentionally bringing its window to
the foreground. While ChatGPT responds, the editor switches to a read-only
response view, its action buttons and the microphone are hidden, and the status
label shows activities such as web searching when ChatGPT exposes them in the turn.
The reply streams into the former input area. After the completed reply becomes
stable, Live GPT clicks ChatGPT's **Read aloud** button. During playback, the
response area becomes a large, full-width two-label subtitle view that follows
the browser media's playback progress and rolls forward one line at a time. If ChatGPT
does not expose its media element, subtitle timing falls back to an estimate.
The full response and microphone return when playback finishes. Click the
completed reply or press the microphone to start another prompt. Press
**Enter** to send a prompt or **Shift+Enter** to insert a newline. Both editor
buttons remain hidden while the editor is empty.

At startup, Live GPT connects through Playwright to a local Chromium browser
that was started with remote debugging enabled. It discovers ChatGPT tabs and
lists them by page title beside the **Live GPT** heading, refreshing the list as
tabs open, close, or navigate. When no debuggable browser is available, Live GPT
opens the current user's `edge://inspect/#remote-debugging` or
`chrome://inspect/#remote-debugging` settings page. Live GPT turns on **Allow
remote debugging for this browser instance** through Windows accessibility,
verifies that the browser's per-user `DevToolsActivePort` marker contains a live
endpoint, and then connects. If the browser is already open, the settings page
opens in a new tab without replacing the active page. The browser can still ask
the user to approve the incoming connection; Live GPT makes one request and
waits for **Enable Debugging** to be clicked before retrying a declined or timed
out request. When a live debugging marker already exists, that button retries
the connection directly and leaves the browser tabs unchanged. Attached
ChatGPT pages retain the browser's native color-scheme
preference. Live GPT does not launch the browser with debugging flags or create
a separate profile. The optional `LIVE_GPT_CDP_ENDPOINT` environment variable
can point the app at a specific CDP HTTP or WebSocket endpoint.
