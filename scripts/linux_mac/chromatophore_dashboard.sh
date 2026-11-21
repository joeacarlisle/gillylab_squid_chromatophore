#!/bin/bash
#
# Make executable once with:  chmod +x chromatophore_dashboard.sh

# Move to the project root (up two levels from where this script is located)
cd "$(dirname "$0")/../.."

if [ -f ".venv/bin/python" ]; then
    echo "[INFO] Found virtual environment. Launching..."
    PYTHON_EXE=".venv/bin/python"
else
    echo "[INFO] Virtual environment not found. Attempting to use global Python..."
    PYTHON_EXE="python3"
fi

# First start: the dashboard needs the database that the analysis writes. Build it from the videos in videos/.
if [ ! -f "processed/gl_chromatophore.duckdb" ]; then
    echo "[INFO] First start: no database yet. Analysing the videos in videos/ (a minute or two)..."
    if ! "$PYTHON_EXE" main.py --demo; then
        echo
        echo "[ERROR] The analysis failed. Please check the messages above."
        read -p "Press any key to continue..."
        exit 1
    fi
fi

# --static-dirs lets the browser play videos/*.mp4.  Remove --show on headless servers.
"$PYTHON_EXE" -m panel serve chromatophore_holoviews.py --port 5009 --static-dirs videos=videos --show
if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] The dashboard failed to launch. Please check the logs above."
    read -p "Press any key to continue..."
fi
