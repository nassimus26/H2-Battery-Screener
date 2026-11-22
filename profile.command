#!/bin/bash
cd "$(dirname "$0")"
pkill -f "scripts/dev/profile_gas.py" 2>/dev/null; sleep 1
.venv/bin/python "scripts/dev/profile_gas.py" > scripts/dev/profile.log 2>&1
echo "exit $?" >> scripts/dev/profile.log
(sleep 1; osascript -e 'tell application "Terminal" to close (every window whose name contains "profile")' >/dev/null 2>&1) &
exit 0
