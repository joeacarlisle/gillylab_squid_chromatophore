@echo off
:: Move to the project root (up two levels from where this script is located)
pushd "%~dp0..\.."

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Found virtual environment. Launching...
    set PYTHON_EXE=.venv\Scripts\python.exe
) else (
    echo [INFO] Virtual environment not found. Attempting to use global Python...
    set PYTHON_EXE=python
)

:: First start: the dashboard needs the database that the analysis writes. Build it from the videos in videos\.
if not exist "processed\gl_chromatophore.duckdb" (
    echo [INFO] First start: no database yet. Analysing the videos in videos\ ^(a minute or two^)...
    %PYTHON_EXE% main.py --demo
    if errorlevel 1 (
        echo.
        echo [ERROR] The analysis failed. Please check the messages above.
        pause
        popd
        exit /b 1
    )
)

:: --static-dirs lets the browser play videos\*.mp4
%PYTHON_EXE% -m panel serve chromatophore_holoviews.py --port 5009 --static-dirs videos=videos --show

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The dashboard failed to launch. Please check the logs above.
    pause
)
popd
