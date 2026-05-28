$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$global:GMKT_REPO_ROOT = $repoRoot
$global:GMKT_VENV_PYTHON = Join-Path $repoRoot ".venv\Scripts\python.exe"
$global:GMKT_TOOL_SCRIPT = Join-Path $repoRoot "tools\cli.py"

function global:gmkt {
    if (Test-Path $global:GMKT_VENV_PYTHON) {
        & $global:GMKT_VENV_PYTHON $global:GMKT_TOOL_SCRIPT @args
    }
    else {
        python $global:GMKT_TOOL_SCRIPT @args
    }
}

Write-Host "Current session now supports: gmkt <subcommand> ..."
Write-Host "Example: gmkt help"
Write-Host "This only affects the current PowerShell session."

if ($args.Count -gt 0) {
    gmkt @args
}
