<#
.SYNOPSIS
    Launches the "iPad as camera + interface" showcase for the elderly-care
    monitoring project.

.DESCRIPTION
    Auto-starts the three processes the showcase needs. When Windows
    Terminal (wt) is installed, all of them run as tabs in one wt window;
    otherwise each falls back to its own separate pwsh window:

      relay        python relay\server.py
      cloudflared  quick tunnel to http://localhost:10000
      app          python main.py --source ipad ...

    Uses the SYSTEM Python interpreter (not .venv): requirements-ipad.txt
    documents that .venv on this machine is a stale, unused environment
    missing aiortc/cv2/mediapipe/torch entirely, while system Python (on
    PATH) has the full working set main.py actually needs. This launcher
    resolves and hardcodes that interpreter's full path up front so it
    never depends on PATH ordering inside a spawned shell (Windows also
    ships a WindowsApps python.exe stub that must not win that race).

    RELAY_SECRET and IPAD_ROOM are read from .env at the repo root, trimmed
    of surrounding whitespace/quotes, and set only into this session's
    environment ($env:...). Child processes started with Start-Process
    inherit that environment, so the secret never appears on a command
    line, in shell history, or in a log file, and .env itself is never
    modified.

    cloudflared is started with --logfile so this launcher can scrape the
    quick-tunnel URL (https://<random>.trycloudflare.com) out of the log
    even though cloudflared runs in its own separate tab/window. That URL is
    then passed to main.py via --ipad-relay-url, so the operator never has
    to hand-edit IPAD_RELAY_URL in .env.

.USAGE
    pwsh -File scripts\start-showcase.ps1
    pwsh -File scripts\start-showcase.ps1 --no-ipad-toggle
    (Any extra arguments are forwarded verbatim to main.py.)
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# 1. Resolve repo root and verify prerequisites.
# ---------------------------------------------------------------------------
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$envFile = Join-Path $root '.env'

if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    Write-Error "Missing .env at repo root: $envFile"
    exit 1
}

# System Python, not .venv: requirements-ipad.txt records that .venv on this
# machine is stale and missing aiortc/cv2/mediapipe/torch, while system
# Python (first on PATH) has the working dependency set main.py needs.
# Resolved once here to a full path so every spawned window uses the exact
# same interpreter regardless of that shell's own PATH ordering.
$pythonCmd = $null
try {
    $pythonCmd = (Get-Command python -ErrorAction Stop).Source
}
catch {
    $pythonCmd = $null
}
if (-not $pythonCmd -or $pythonCmd -like '*WindowsApps*') {
    Write-Error ("No working system Python found on PATH (found: " +
        "$pythonCmd). Install Python 3.12 with the app's dependencies " +
        "(see requirements.txt / requirements-ipad.txt) and ensure it " +
        "precedes the WindowsApps python.exe stub on PATH.")
    exit 1
}
$pythonExe = $pythonCmd

$cloudflaredCmd = $null
try {
    $cloudflaredCmd = Get-Command cloudflared -ErrorAction Stop
}
catch {
    $cloudflaredCmd = $null
}
if (-not $cloudflaredCmd) {
    Write-Error "cloudflared was not found on PATH. Install it or add it to PATH before running this launcher."
    exit 1
}

# Windows Terminal, if present, lets every process live as a tab in one
# window instead of spawning four separate top-level pwsh windows. Falls
# back to the old one-window-per-process behavior when wt isn't installed.
$wtCmd = $null
try {
    $wtCmd = Get-Command wt -ErrorAction Stop
}
catch {
    $wtCmd = $null
}
$useWt = [bool]$wtCmd

# Opens one or more tabs in the showcase's Windows Terminal window. Each
# entry is a @{ Title; Command } hashtable. '-w 0' targets "the most
# recently used wt window" - the first call (no such window yet) creates
# one, every later call reuses it, so all tabs land together.
function Start-WtTabs {
    param(
        [Parameter(Mandatory)] [array] $Tabs
    )
    $wtArgs = @('-w', '0')
    for ($i = 0; $i -lt $Tabs.Count; $i++) {
        if ($i -gt 0) { $wtArgs += ';' }
        $wtArgs += @('new-tab', '-d', $root, '--title', $Tabs[$i].Title, 'pwsh', '-NoExit', '-Command', $Tabs[$i].Command)
    }
    Start-Process wt -ArgumentList $wtArgs
}

# ---------------------------------------------------------------------------
# 2. Parse RELAY_SECRET and IPAD_ROOM out of .env (without loading the whole
#    file into the process environment, and without ever printing the
#    secret).
# ---------------------------------------------------------------------------
function Get-EnvValue {
    param(
        # Not Mandatory: PowerShell's Mandatory-parameter binder rejects an
        # ENTIRE [string[]] array if any single element is "" -- and a .env
        # file with blank separator lines (completely normal) triggers that
        # on every call, before a single line is even inspected. Handling
        # $null/empty explicitly below is the correct guard instead.
        [string[]] $Lines,
        [Parameter(Mandatory)] [string] $Name
    )
    if (-not $Lines) { return $null }

    foreach ($line in $Lines) {
        $trimmedLine = $line.Trim()
        if ($trimmedLine.Length -eq 0 -or $trimmedLine.StartsWith('#')) {
            continue
        }
        if ($trimmedLine -notmatch '^([^=]+)=(.*)$') {
            continue
        }
        $key = $Matches[1].Trim()
        if ($key -ne $Name) {
            continue
        }
        $value = $Matches[2].Trim()
        # Strip one layer of surrounding matching quotes, single or double.
        if ($value.Length -ge 2) {
            $first = $value.Substring(0, 1)
            $last = $value.Substring($value.Length - 1, 1)
            if (($first -eq '"' -and $last -eq '"') -or ($first -eq "'" -and $last -eq "'")) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }
        return $value
    }
    return $null
}

$envLines = Get-Content -LiteralPath $envFile
$relaySecret = Get-EnvValue -Lines $envLines -Name 'RELAY_SECRET'
$ipadRoom = Get-EnvValue -Lines $envLines -Name 'IPAD_ROOM'

if ([string]::IsNullOrWhiteSpace($relaySecret)) {
    Write-Error "RELAY_SECRET is missing or empty in .env"
    exit 1
}
if ([string]::IsNullOrWhiteSpace($ipadRoom)) {
    Write-Error "IPAD_ROOM is missing or empty in .env"
    exit 1
}

# Set into this session only. Start-Process below inherits these via the
# current process environment - they are never written back to .env and
# never appear on a command line.
$env:RELAY_SECRET = $relaySecret
$env:IPAD_ROOM = $ipadRoom

Write-Host "Repo root:   $root"
Write-Host "Python:      $pythonExe"
Write-Host "RELAY_SECRET: (loaded, not shown)"
Write-Host "IPAD_ROOM:    $env:IPAD_ROOM"
Write-Host ""

# ---------------------------------------------------------------------------
# 3-4. Terminal 1 (relay server) + Terminal 2 (cloudflared quick tunnel),
#      logging to a temp file so we can scrape the assigned URL even though
#      it runs in its own tab. Started together as two tabs in one wt window
#      when Windows Terminal is available; otherwise as two separate windows.
# ---------------------------------------------------------------------------
$log = Join-Path $env:TEMP ("cloudflared-showcase-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
if (Test-Path -LiteralPath $log) {
    Remove-Item -LiteralPath $log -Force
}

$relayCmd = "`"$pythonExe`" relay\server.py"
$cloudflaredCmdLine = "cloudflared tunnel --url http://localhost:10000 --logfile `"$log`""

if ($useWt) {
    Write-Host "Starting relay server + cloudflared tunnel tabs..."
    Write-Host "  Log file: $log"
    Start-WtTabs -Tabs @(
        @{ Title = 'relay'; Command = $relayCmd },
        @{ Title = 'cloudflared'; Command = $cloudflaredCmdLine }
    )
}
else {
    Write-Host "Starting Terminal 1 (relay server)..."
    Start-Process pwsh -ArgumentList '-NoExit', '-Command', $relayCmd -WorkingDirectory $root

    Write-Host "Starting Terminal 2 (cloudflared quick tunnel)..."
    Write-Host "  Log file: $log"
    Start-Process pwsh -ArgumentList '-NoExit', '-Command', $cloudflaredCmdLine -WorkingDirectory $root
}

# ---------------------------------------------------------------------------
# 5. Poll the log file for the quick-tunnel URL.
# ---------------------------------------------------------------------------
Write-Host "Waiting for cloudflared to report the tunnel URL..."
$urlPattern = 'https://[-a-z0-9]+\.trycloudflare\.com'
$relayUrl = $null

for ($i = 0; $i -lt 80; $i++) {
    Start-Sleep -Milliseconds 500
    $content = Get-Content -LiteralPath $log -Raw -ErrorAction SilentlyContinue
    if ([string]::IsNullOrEmpty($content)) {
        continue
    }
    $match = [regex]::Match($content, $urlPattern)
    if ($match.Success) {
        $relayUrl = $match.Value
        break
    }
}

if (-not $relayUrl) {
    Write-Error "cloudflared did not report a tunnel URL within the timeout. Check the cloudflared tab/window and the log file: $log"
    exit 1
}

Write-Host "Tunnel URL: $relayUrl"
Write-Host ""

# ---------------------------------------------------------------------------
# 6. Terminal 3 - the app, pointed at the freshly captured tunnel URL. Any
#    extra arguments passed to this launcher are forwarded to main.py.
# ---------------------------------------------------------------------------
$extraArgs = $args -join ' '
$mainCmd = "`"$pythonExe`" main.py --source ipad --webui --enable-multi-person --ipad-relay-url `"$relayUrl`" --ipad-room `"$env:IPAD_ROOM`""
if ($extraArgs.Length -gt 0) {
    $mainCmd = "$mainCmd $extraArgs"
}

# ---------------------------------------------------------------------------
# 6b. iPad Pairing tab/window - shows a scannable QR code for the pairing
#     URL, so the operator can scan it with the iPad camera instead of
#     typing the random trycloudflare URL. Best effort: scripts\show_qr.py
#     degrades to printing the plain URL if 'qrcode' isn't installed.
# ---------------------------------------------------------------------------
$pairingUrl = "$relayUrl/r/$env:IPAD_ROOM"
$qrCmd = "`"$pythonExe`" scripts\show_qr.py `"$pairingUrl`""

if ($useWt) {
    Write-Host "Starting app + iPad pairing (QR code) tabs..."
    Start-WtTabs -Tabs @(
        @{ Title = 'app'; Command = $mainCmd },
        @{ Title = 'pairing'; Command = $qrCmd }
    )
}
else {
    Write-Host "Starting Terminal 3 (app)..."
    Start-Process pwsh -ArgumentList '-NoExit', '-Command', $mainCmd -WorkingDirectory $root

    Write-Host "Starting iPad Pairing window (QR code)..."
    Start-Process pwsh -ArgumentList '-NoExit', '-Command', $qrCmd -WorkingDirectory $root
}

# ---------------------------------------------------------------------------
# 7. Summary for the operator.
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "=================================================================="
Write-Host " Showcase launched"
Write-Host "=================================================================="
Write-Host " Tunnel URL:        $relayUrl"
Write-Host ""
Write-Host " iPad pairing URL:  $pairingUrl"
Write-Host ""
Write-Host " The 6-digit pairing code appears in the app tab/window."
Write-Host " Keep the relay tab/window running for the whole session."
Write-Host "=================================================================="
