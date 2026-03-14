@echo off
setlocal
set "REPO_ROOT=%~dp0"
set "VENV_PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
set "TOOL_SCRIPT=%REPO_ROOT%tools\cli.py"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%TOOL_SCRIPT%" %*
) else (
    python "%TOOL_SCRIPT%" %*
)

