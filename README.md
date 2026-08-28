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
