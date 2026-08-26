<#
.SYNOPSIS
    Run the CareVision feature tests that back the final presentation, green.

.DESCRIPTION
    Runs the narrated showcase plus the existing on-topic test files (the
    conversational agent, alerts, voice/showcase, iPad protocol, skin screen)
    as a single pytest session and prints a pass/fail summary. Auto-selects the
    interpreter that can import the stack (prefers one with aiortc for the iPad
    tests). Read-only: it runs tests, it changes nothing.

.EXAMPLE
    pwsh -File scripts/run-presentation-tests.ps1
#>
[CmdletBinding()]
param(
    [switch]$ShowcaseOnly   # run only the narrated showcase wrapper
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Pick an interpreter that imports the vision + iPad stack. System python first
# (the checked-in .venv is missing aiortc, which the iPad protocol tests need).
$candidates = @("python", (Join-Path $repo ".venv\Scripts\python.exe"))
$py = $null
foreach ($c in $candidates) {
    try {
        & $c -c "import numpy, yaml" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $c; break }
    } catch { }
}
if (-not $py) {
    Write-Error "No interpreter could import numpy + yaml. Install requirements first."
    exit 2
}
Write-Host "[tests] interpreter: $py" -ForegroundColor Cyan

# The narrated showcase always runs; the rest are the existing on-topic suites.
$tests = @("tests/presentation_showcase_test.py")
if (-not $ShowcaseOnly) {
    $tests += @(
        "tests/corroboration_and_elicitation_test.py",
        "tests/corroboration_airlock_test.py",
        "tests/corroboration_steering_test.py",
        "tests/answer_classifier_test.py",
        "tests/async_answer_classification_test.py",
        "tests/adaptive_question_test.py",
        "tests/topic_table_test.py",
        "tests/conversation_agent_test.py",
        "tests/multi_person_detection_test.py",
        "tests/module_toggle_test.py",
        "tests/tts_and_showcase_test.py",
        "tests/showcase_gate_test.py",
        "tests/ipad_protocol_test.py",
        "tests/ipad_link_defense_test.py",
        "tests/vlm_cues_test.py"
    )
}

# Keep only the tests that actually exist, so a renamed file fails loudly rather
# than silently skewing the count.
$present = $tests | Where-Object { Test-Path (Join-Path $repo $_) }
$missing = $tests | Where-Object { -not (Test-Path (Join-Path $repo $_)) }
foreach ($m in $missing) { Write-Warning "skipping (not found): $m" }

Write-Host "[tests] running $($present.Count) suites via pytest ..." -ForegroundColor Cyan
& $py -m pytest -q @present
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "[tests] ALL GREEN - presentation suites pass." -ForegroundColor Green
} else {
    Write-Host "[tests] FAILURES (pytest exit $code) - see output above." -ForegroundColor Red
}
exit $code
