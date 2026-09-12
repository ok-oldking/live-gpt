# live-gpt
voice and share screen chat with your web chatbot, free of tokens

## Run

```powershell
.\.venv\Scripts\Activate.ps1
live-gpt
```

The app starts as a translucent, always-on-top overlay centered above the
bottom of the screen and also places an icon in the Windows system tray. Drag
any non-interactive area of the overlay to reposition it, or drag any border or corner to resize
it. The lock button beside **Configure** freezes both its position and size;
click it again to unlock. Screen pointer tracking keeps the overlay visible within a 5% margin on each
edge. Outside that boundary, everything except the pet becomes transparent,
including the input, subtitles, and dictation area. The overlay stays visible
while dragging or resizing. The **Auto-hide** button hides the overlay completely so clicks pass
through to the underlying app. In auto-hide mode it reappears for dictation,
incoming replies, and Read Aloud, then hides five seconds after playback ends.
If dictation produces text, the overlay stays visible so that text can be
reviewed or sent; enabling auto-hide does not hide unsent text or interrupt
dictation, response generation, playback, or an expanded subtitle. Text recorded with the UI microphone stays in the input until you explicitly
send it. Recording hotkeys automatically send successful dictation. Voice recognition must produce
at least two non-whitespace characters before it can be sent. A one-character
result remains in the editor for correction; manually typed one-character
messages are still allowed.
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
least 0.5 seconds after it is pressed or it is cancelled. The global
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
changing the selector. For dictated prompts, Live GPT
captures and pastes the selected screenshot into ChatGPT after the microphone
has remained pressed for 0.5 seconds. A cancelled short press uploads nothing.
Starting another recording before the pending prompt is sent removes the
previously uploaded screenshot before preparing the new one.

The **Configure** button opens the application settings, organized into
Shortcuts, Language, Recording, and Playing pages in the left navigation. English and
Chinese can be selected as a preview, but the selection does not change the UI
yet. Every valid change is saved immediately, so the styled title-bar close
button is the only dismissal control and there is no separate Save Changes step.
Recording independently chooses the browser or local Sherpa-ONNX, while
Playing independently chooses browser playback or an existing
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) installation with its
embedded Python runtime. Local controls stay hidden when browser playback is selected.
Old Qwen3 and CosyVoice playback selections fall back to browser playback.

For GPT-SoVITS, select the installation folder and reference audio, and optionally
set the reference transcript and reference/output languages. You can provide a
GPT `.ckpt` model and a SoVITS `.pth` model. Both paths must point to existing
files, or both must be blank to use the installation's configured models.
Incomplete pairs are not saved and cannot be checked or played. Changing the
pair restarts the local server before the next synthesis; clearing both restores
the installation defaults.

Live GPT copies `sovits_server.py` into the selected installation and runs it with
`runtime/python.exe`. It preloads the model, reuses reference-audio features, and
streams speech through one audio output. Complete sentences enter the synthesis
queue while ChatGPT is still generating its reply.

The STT catalog includes Mandarin and English models. Recording settings can
install the Sherpa runtime and download models into the working directory's
`models` folder. Installation supports official PyPI, Aliyun, and SJTUG without
changing global pip settings. **Record microphone** tests transcription;
**Play text** tests GPT-SoVITS playback.

Hotkey defaults are
**Right Alt** (record and send with the selected screenshot) and
**Right Ctrl** (record and send without a screenshot).
These are the two global shortcuts, and both are enabled by default.
Click a shortcut field and press a key or combination to change it;
left and right Alt, Ctrl, and Shift are recognized separately.
Use the **×** inside a shortcut field to clear it and disable that action.
Cleared shortcuts stay empty after restarting; changes are saved automatically.
Recording shortcuts automatically send successful dictation of at least two
characters; the UI microphone always requires manual sending. Starting either
recording shortcut stops active local or
browser voice playback. The shortcuts work while the
overlay is hidden or unfocused. Live GPT observes their key state without
registering or swallowing the keys, so the foreground program continues to
receive the same keystrokes. Changes are saved for the current Windows user.

Preferences are stored as JSON in `%APPDATA%\Live GPT\config.json`. The file is
updated automatically when a preference changes and includes the language,
hotkeys, auto-hide state, selected screenshot source, selected
ChatGPT conversation, independent recording/playback backends, local STT/TTS
models, GPT-SoVITS installation, paired model paths,
languages, reference-audio path and transcript, overlay position
and size, and
position-lock state. A
missing, unreadable, or invalid configuration falls back to safe defaults and
is repaired on disk. Existing registry-based hotkeys are migrated on the first
JSON-configured launch.
The editor and microphone remain disabled until a ChatGPT window is connected.
Status tips appear as the editor hint rather than in a separate row; errors use
a red hint.

Sending switches directly to a full-width two-label subtitle view. While
ChatGPT responds, its first line shows the sent prompt and its second line
shows the current status. When **Read aloud** starts, the prompt is removed and
both lines show the response. During playback, subtitles follow the browser
media's progress and roll forward one line at a time. Hover over the subtitle
area to expand a full-response view whose height fits the wrapped text. Moving
the pointer outside the whole overlay restores the two-line view at the current playback
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
the user to approve the incoming connection. Live GPT keeps one approval request
pending until the user chooses **Allow** or **Deny**, preventing unanswered
requests from creating a stack of dialogs. After choosing **Deny**, click
**Enable Debugging** to make another request; the browser tabs remain unchanged.
Attached
ChatGPT pages retain the browser's native color-scheme
preference. Live GPT does not launch the browser with debugging flags or create
a separate profile. The optional `LIVE_GPT_CDP_ENDPOINT` environment variable
can point the app at a specific CDP HTTP or WebSocket endpoint.

## Pet

The default pet is `assets/pets/feibi-jiubi`. Open **Settings → Pet** to choose
a pet from the folders under `assets/pets`; changes apply immediately.
Idle animation can always play, stay on a still pose, or play for a duration
(default 10 seconds) each time the pet returns to idle. Looking and activity
animations are unaffected. Settings are saved automatically. You can also set
`pet_path` in the user configuration to an external pet directory or `pet.json`
(restart after manual configuration edits). An empty path uses the bundled default. Both v1 (8×9)
and v2 (8×11) sprite sheets with 192×208 cells are supported; an omitted
`spriteVersionNumber` is treated as v1. Invalid custom pets fall back to the default.

The pet uses idle at rest, waiting during dictation, running while awaiting a
response, review during speech playback, and failed for errors. Dragging uses
running-right or running-left and restores the current activity on release.
V2 pets look toward the screen pointer while idle when its distance from the
pet is less than half the height of the screen containing the pet. Moving
farther away or leaving the pointer still for five seconds restores the idle
loop. The lock button locks overlay dragging and resizing.

Mouse-out hiding and auto-hide only run while a ChatGPT browser tab is connected.
Before connection or after disconnection, the full overlay stays visible so you
can reconnect. Your auto-hide preference is retained for the next connection.

Dragging keeps the overlay inside the usable area of the screen under the
pointer. Move the pointer onto another monitor to transfer the overlay there.
If the overlay is larger than that screen's usable area, its top-left corner
stays on-screen so its controls remain reachable.

Click a response to edit, select, copy, or clear it without losing its text.
HTTP(S) links open in the default browser. Holding the microphone for at least
0.5 seconds clears the previous input; shorter taps preserve it.

Hover anywhere over the overlay to expand the input vertically to fit its
content. Expansion is limited to the screen's usable height, with scrolling
for longer text. Moving away restores the compact window size.

## Download pets

In **Configure → Pet**, paste a public GitHub folder URL (for example,
`https://github.com/legeling/awesome-codex-pet/tree/main/pets/citlali--zaytsevzy`)
and click **Download pet**. The folder must contain `pet.json` and
`spritesheet.webp`. Both `/tree/` and `/blob/` folder URLs are supported.
Find pet folder links in [awesome-codex-pet](https://github.com/legeling/awesome-codex-pet).
After validation, the pet is saved under
`download/pets/<folder-name>`, added to the list, and selected immediately.
Downloaded pets remain available after restarting. Existing valid downloads
are reused; failed downloads leave the selected pet unchanged.
