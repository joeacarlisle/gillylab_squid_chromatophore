#!/usr/bin/env bash
# Runs the setup.sh in the project root (this script can be started from anywhere).
exec "$(dirname "$0")/../../setup.sh" "$@"
