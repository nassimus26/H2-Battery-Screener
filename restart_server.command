#!/bin/bash
# Stop the running simulator (if any) and start it again with the latest code.
# The server runs in the background (log: scripts/dev/server.log) and this Terminal window closes itself.
cd "$(dirname "$0")"
pkill -f "streamlit run app.py" 2>/dev/null
for i in $(seq 1 30); do                       # wait until the old server has released the port
  lsof -nP -iTCP:8501 -sTCP:LISTEN >/dev/null 2>&1 || break
  sleep 1
done
pkill -9 -f "streamlit run app.py" 2>/dev/null
sleep 1
nohup .venv/bin/streamlit run app.py --server.headless true --server.port 8501 --server.fileWatcherType none \
  > scripts/dev/server.log 2>&1 < /dev/null &
disown
(sleep 1; osascript -e 'tell application "Terminal" to close (every window whose name contains "restart_server")' >/dev/null 2>&1) &
exit 0
