#!/usr/bin/env bash
# Creates the virtual environment for the squid project in this folder and installs it (Linux / macOS).
# Needs Python 3.13. The pinned package versions are in pyproject.toml.
set -e

# Keep the window open at the end (also after an error) so the messages can be read.
finish() {
  code=$?
  echo
  if [ "$code" -ne 0 ]; then echo "Setup failed (exit code $code). See the messages above."; fi
  if [ -t 0 ]; then read -r -p "Press Enter to close this window..." _; fi
}
trap finish EXIT
cd "$(dirname "$0")"

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "Python 3.13 not found."
  echo "  macOS (Homebrew):   brew install python@3.13"
  echo "  Ubuntu / Debian:    see the README (apt or pyenv)"
  exit 1
fi

if [ ! -d .venv ]; then
  python3.13 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

echo
echo "Done. Start the dashboard with:  ./scripts/linux_mac/chromatophore_dashboard.sh"
echo "(the first start analyses the demo videos, which takes a minute or two)"
echo "(run 'chmod +x scripts/linux_mac/*.sh' once if they are not executable)"
echo "CHROMAS (single chromatophores) is a separate install: see ../gillylab_chromas/README.md"
