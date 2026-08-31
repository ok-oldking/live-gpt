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
resized. The **Auto-hide** button hides the overlay completely so clicks pass
through to the underlying app. In auto-hide mode it reappears for dictation,
incoming replies, and Read Aloud, then hides five seconds after playback ends.
If dictation produces text, the overlay stays visible so that text can be
reviewed or sent; enabling auto-hide does not hide unsent text or interrupt
dictation, response generation, playback, or an expanded subtitle. Enable
**Auto Send** to send successful dictation
immediately, using the selected screenshot when one is selected and sending
without one when **No screenshot** is selected.
Double-click the tray icon to reveal it and turn auto-hide off. The **Exit**
button closes the app.

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

The recognized text remains editable. Its paper-plane button sends the current
text through Playwright to the selected
ChatGPT tab, and its **X** button clears the editor. The app fills ChatGPT's
composer and clicks its send button, then clears the local text after ChatGPT
accepts it. The **X** button also clears the selected ChatGPT tab's composer.
These browser interactions run without intentionally bringing its window to
the foreground. A send initiated inside the overlay restores focus to the
previous app; a global-hotkey send leaves focus unchanged.

The screenshot selector starts with **No screenshot**, followed by one choice
for each desktop display and then visible windows ordered from largest to
smallest. A window is listed only when its area is greater than one eighth of
the display containing it. When sending, Live GPT removes existing ChatGPT
image attachments and, unless **No screenshot** is selected, captures the
chosen source, scales its longest edge down to 1920 pixels when necessary, and
uploads it as lossless WebP before clicking Send. Window capture uses the
BitBlt/PrintWindow approach with full-content rendering enabled.
When a screenshot source is selected, the **With Screenshot** action remains
available even with an empty prompt. If the editor also contains text, a
paper-plane **No Screenshot** action appears so the prompt can be sent without
changing the selector.

The **Configure** button opens the application settings, organized into global
shortcut and interface-language pages in the left navigation. English and
Chinese can be selected as a preview, but the selection does not change the UI
yet. Hotkey defaults are
**Caps Lock** (hold to dictate), **Ctrl+S** (send with the selected screenshot),
and **Ctrl+D** (send without a screenshot). The shortcuts work while the
overlay is hidden or unfocused. Live GPT observes their key state without
registering or swallowing the keys, so the foreground program continues to
receive the same keystrokes. Changes are saved for the current Windows user.
The editor and microphone remain disabled until a ChatGPT window is connected.
Status tips appear as the editor hint rather than in a separate row; errors use
a red hint.

Sending switches directly to a full-width two-label subtitle view. While
ChatGPT responds, its first line shows the sent prompt and its second line
shows the current status. When **Read aloud** starts, the prompt is removed and
both lines show the response. During playback, subtitles follow the browser
media's progress and roll forward one line at a time. Hover over the subtitle
area to expand a full-response view whose height fits the wrapped text. Moving
the pointer outside always restores the two-line view at the current playback
line. Left-click it to return to input mode. If
ChatGPT does not expose its media element, subtitle timing falls back to an
estimate. Press the microphone or its hotkey to dismiss subtitles and dictate
another prompt. Press
**Enter** to send a prompt or **Shift+Enter** to insert a newline. With no
screenshot selected, the send actions remain hidden while the editor is empty.

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
the user to approve the incoming connection. If approval arrives after a
connection request times out, Live GPT retries automatically at a short interval
until it connects. When a live debugging marker already exists, **Enable
Debugging** retries immediately and leaves the browser tabs unchanged. Attached
ChatGPT pages retain the browser's native color-scheme
preference. Live GPT does not launch the browser with debugging flags or create
a separate profile. The optional `LIVE_GPT_CDP_ENDPOINT` environment variable
can point the app at a specific CDP HTTP or WebSocket endpoint.
