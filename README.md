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
without one when **No screenshot** is selected. Voice recognition must produce
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
changing the selector. For dictated prompts using **Auto Send**, Live GPT
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
Playing independently chooses the browser or local
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), or local
[CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice), or an existing
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) installation with its
embedded Python runtime. Local controls
stay hidden when the corresponding web engine is selected. The STT catalog
includes five Mandarin and five English options. Playback offers Qwen3-TTS
CustomVoice 0.6B and 1.7B models with nine named Chinese, English, Japanese,
and Korean speakers. CosyVoice offers the Fun-CosyVoice3 0.5B 2512 streaming
model with zero-shot voice cloning: choose a reference WAV and enter its exact
transcript, or leave both blank to use the official example voice. Local Qwen
and CosyVoice playback require a detected NVIDIA GPU and are
blocked when CUDA-enabled PyTorch is unavailable. Runtime repair installs a
matched CUDA 12.6 PyTorch/Torchaudio build from PyTorch's NVIDIA wheel index;
the selected PyPI mirror is added as the dependency index for that installation.
Repair preserves matching PyTorch files so Windows never needs to overwrite a
CUDA extension currently loaded by the running app; only missing or mismatched
packages are changed.
GPT-SoVITS is not installed or modified by Live GPT apart from copying
`sovits_server.py` into the selected installation. Live GPT launches that file
with `runtime/python.exe`, warms the configured model once, reuses GPT-SoVITS's
reference-audio feature cache, and consumes native streaming audio. Its text
language is configurable; reference audio, reference transcript, and prompt
language are saved as optional settings (reference audio is required when
synthesis is requested).
Qwen playback generates a first natural text chunk immediately, then uses a
two-chunk lookahead buffer while later chunks are generated concurrently with
current audio playback. Deterministic decoding keeps the speaker timbre steadier;
generated edge silence is trimmed and short fades soften chunk joins. Playback
language can be selected explicitly from Qwen's ten languages or left on the
saved Auto default. The selected local model is preloaded in a background thread
at startup so model initialization is normally complete before the first reply
is played. Invalid or incomplete runtimes are never preloaded, leaving their
binary files unlocked so Install / Repair can recover them. CUDA loading enables BF16/FP16, optimized SDPA
kernels, and TF32 where applicable. Install / Repair also installs the official

For every local TTS backend, Live GPT sends the first complete sentence to the
model while ChatGPT is still generating. Later complete sentences are added to
the same synthesis queue, generated ahead of playback, and written through one
persistent audio stream until the final response remainder has played.

Install / Repair also installs the official
FlashAttention build prerequisites, attempts `flash-attn --no-build-isolation`
with four build jobs, retries official PyPI when the selected mirror cannot
provide the source package, extracts the unusually deep source tree through a
temporary drive-root or mapped-drive path to avoid the Windows 260-character
limit, verifies the PyPI archive's SHA-256 digest, and omits the AMD-only
Composable Kernel subtree before the NVIDIA build. On Windows it installs and
locates NVIDIA's CUDA 12.6 NVCC, runtime, and CCCL wheels for the build. The
selected PyPI mirror is used first, with official PyPI as fallback. Successful
repair also removes known
invalid `~orch`-style pip backups left by an interrupted reinstall. If the experimental
Windows build is unavailable, the logged error identifies the cause and Qwen
continues with optimized SDPA.
If CPU PyTorch was already loaded by an earlier playback attempt, restart Live
GPT after the repair. It installs each provider's pinned runtime and related
dependencies through the current Python interpreter only when requested,
verifies package versions and wheel integrity, and builds local SHA-256 model
manifests. Qwen and CosyVoice models can be downloaded from Hugging Face (the
default) or ModelScope; each source selection is saved immediately. CosyVoice
repair installs and integrity-checks the official recursive Git checkout,
reuses a valid checkout on later repairs, and safely removes Windows read-only
Git objects from old runtime backups. Models and the CosyVoice runtime are stored under
the working directory's `models` folder and loaded from that local path.
If the selected hub client is missing, Download installs and verifies it through
the selected PyPI mirror before fetching the model. Installer output is
streamed into the app log, with its latest two lines visible beneath setup
progress. Pip's progress stream is forced on for GUI installs and rendered as
a download-description line followed by a live bar with downloaded size,
transfer speed, and ETA. Repeated progress samples update that bottom line in
place. The exact pip and
model-download commands are logged before execution. Active
dependency installs
and model downloads can be cancelled from the settings row; this terminates the
child process and removes temporary model staging files.
The install rows can use official PyPI, Aliyun, or Shanghai Jiao Tong
University's SJTUG mirror without changing the user's global pip configuration;
the mirror choice is shared by Recording and Playing and saved in the app
configuration. Aliyun and SJTUG modes also download CUDA PyTorch from their
dedicated `pytorch-wheels/cu126` mirrors. Aliyun publishes these CUDA wheels
under plain version labels, so the installer uses `2.11.0` there instead of the
`2.11.0+cu126` label used by the official and SJTUG indexes.
**Record
microphone** tests the selected STT model, and **Play text** tests the selected
Qwen model and speaker. Offline and streaming STT models are labeled explicitly.
Streaming recording models decode microphone chunks continuously and update the
test transcript in real time. Each test reports its inference or generation latency.
Hotkey defaults are
**Caps Lock** (record and automatically send with the selected screenshot),
**Shift** (record and automatically send without a screenshot), **Ctrl+S**
(send with the selected screenshot), and **Ctrl+D** (send without a screenshot).
Each shortcut has an **Enabled** switch in settings. Only Caps Lock (Record
and Send with Screenshot) is enabled by default; the other three shortcuts
are disabled until enabled. Switch settings are saved automatically.
The enabled recording shortcuts always send successful dictation regardless of the
Auto Send toggle. Starting either recording shortcut stops active local or
browser voice playback. The shortcuts work while the
overlay is hidden or unfocused. Live GPT observes their key state without
registering or swallowing the keys, so the foreground program continues to
receive the same keystrokes. Changes are saved for the current Windows user.

Preferences are stored as JSON in `%APPDATA%\Live GPT\config.json`. The file is
updated automatically when a preference changes and includes the language,
hotkeys, auto-send and auto-hide states, selected screenshot source, selected
ChatGPT conversation, independent recording/playback backends, local STT/TTS
models, Qwen/CosyVoice download sources, Qwen speaker and playback language,
CosyVoice reference-audio path and transcript, GPT-SoVITS installation,
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
