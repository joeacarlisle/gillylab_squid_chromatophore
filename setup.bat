@echo off
REM Creates the virtual environment for the squid project in this folder and installs it (Windows).
REM Needs Python 3.13. The pinned package versions are in pyproject.toml.
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" not found. Install Python 3.13 with: winget install Python.Python.3.13
  goto fail
)
py -3.13 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.13 not found. Install it with: winget install Python.Python.3.13
  goto fail
)

if not exist .venv (
  py -3.13 -m venv .venv
  if errorlevel 1 goto fail
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
if errorlevel 1 goto fail

echo.
echo Done. Start the dashboard with:  scripts\windows\chromatophore_dashboard.bat
echo ^(the first start analyses the demo videos, which takes a minute or two^)
echo CHROMAS (single chromatophores) is a separate install: see ..\gillylab_chromas\README.md
pause
exit /b 0
:fail
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
