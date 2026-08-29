# live-gpt
voice and share screen chat with your web chatbot, free of tokens

## Run

```powershell
.\.venv\Scripts\Activate.ps1
live-gpt
```

The app starts as a translucent, always-on-top overlay centered above the
bottom of the screen and also places an icon in the Windows system tray. Drag
any non-button area to reposition it, or drag any border or corner to resize
it. The lock button beside **Configure** freezes both its position and size;
click it again to unlock. When the pointer is outside the overlay, its frame,
title controls, and microphone become fully transparent while the input area
stays visible. They return when the pointer enters or while a border is being
resized. Use the overlay's **Hide** and **Exit** buttons, or double-click the
tray icon to restore a hidden overlay.

Press and hold the microphone button to start ChatGPT's browser dictation
through Playwright. Releasing it clicks ChatGPT's **Done** control, waits for
the dictated text to appear in ChatGPT's composer, and copies that text into
the Live GPT input. The ChatGPT tab owns microphone capture and speech
recognition, so the browser may ask for microphone permission the first time.
Live GPT does not record audio locally or download a speech-recognition model.
While the button is held, the input area is replaced by a waiting state until
ChatGPT's dictation-end control appears, followed by a listening state. Closing
the app cancels these browser waits promptly. Before a new session, Live GPT
cancels any stale browser dictation. A mouse dictation must remain held for at
least 0.5 seconds after ChatGPT starts listening or it is cancelled. The global
hold hotkey waits 0.3 seconds before starting, so a quick tap has no effect.

The recognized text remains editable. When the editor contains text, its
paper-plane button sends the current text through Playwright to the selected
ChatGPT tab, and its **X** button clears the editor. The app fills ChatGPT's
composer and clicks its send button, then clears the local text after ChatGPT
accepts it. The **X** button also clears the selected ChatGPT tab's composer.
These browser interactions run without intentionally bringing its window to
the foreground. While ChatGPT responds, the editor switches to a read-only
response view, its action buttons and the microphone are hidden, and the status
label shows activities such as web searching when ChatGPT exposes them in the turn.

The screenshot selector starts with **No screenshot**, followed by one choice
for each desktop display and then visible windows ordered from largest to
smallest. A window is listed only when its area is greater than one eighth of
the display containing it. When sending, Live GPT removes existing ChatGPT
image attachments and, unless **No screenshot** is selected, captures the
chosen source, scales its longest edge down to 1920 pixels when necessary, and
uploads it as lossless WebP before clicking Send. Window capture uses the
BitBlt/PrintWindow approach with full-content rendering enabled.
When a screenshot source is selected, a **No screenshot** action also appears
inside the editor so the current prompt can be sent as text without changing
the selector.

The **Configure** button opens the global-hotkey settings. Defaults are
**Caps Lock** (hold to dictate), **Ctrl+S** (send with the selected screenshot),
and **Ctrl+D** (send without a screenshot). The shortcuts work while the
overlay is hidden or unfocused. Live GPT observes their key state without
registering or swallowing the keys, so the foreground program continues to
receive the same keystrokes. Changes are saved for the current Windows user.
The editor and microphone remain disabled until a ChatGPT window is connected.
Status tips appear as the editor hint rather than in a separate row; errors use
a red hint.

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
