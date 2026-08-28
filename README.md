# live-gpt
voice and share screen chat with your web chatbot, free of tokens

## Run

```powershell
.\.venv\Scripts\Activate.ps1
live-gpt
```

The app starts as a translucent, always-on-top overlay centered above the
bottom of the screen and also places an icon in the Windows system tray. Drag
any non-button area to reposition it. Use the overlay's **Settings**, **Hide**,
and **Exit** buttons, or double-click the tray icon to restore a hidden overlay.
Press and hold the microphone button to capture audio; releasing it saves a
WAV file in the `recordings` directory.
Use **Settings** to choose the recording and playback devices, or leave either
one on the first device marked **(System Default)**.
Recording prefers 16 kHz and automatically falls back to the selected
microphone's native sample rate when required by Windows.

Live transcription uses the small bilingual Chinese-English streaming
Zipformer model from sherpa-onnx. The model is downloaded into `models/` on
first launch, then reused offline. Partial and final recognition results appear
in the overlay while the microphone button is held.

The recognized text remains editable. When the editor contains text, its
paper-plane button submits the current text through the app's `send_requested`
signal, and its **X** button clears the editor. Both buttons remain hidden while
the editor is empty.

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
out request. Attached ChatGPT pages retain the browser's native color-scheme
preference. Live GPT does not launch the browser with debugging flags or create
a separate profile. The optional `LIVE_GPT_CDP_ENDPOINT` environment variable
can point the app at a specific CDP HTTP or WebSocket endpoint.
