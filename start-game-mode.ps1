# Match the privilege level of games launched as administrator so Windows
# allows Live GPT's global microphone shortcuts to read their key state.
$ErrorActionPreference = 'Stop'
$gamePython = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
$gameEntry = Join-Path $PSScriptRoot 'main.py'

if (-not (Test-Path -LiteralPath $gamePython -PathType Leaf)) {
    throw 'Live GPT virtual environment is missing. Install the project in .venv first.'
}

try {
    Start-Process -FilePath $gamePython -ArgumentList ('"{0}"' -f $gameEntry) `
        -WorkingDirectory $PSScriptRoot -Verb RunAs -WindowStyle Hidden -ErrorAction Stop
} catch {
    throw "Could not start Live GPT as administrator. Accept the Windows permission prompt to enable shortcuts over elevated games. $($_.Exception.Message)"
}
