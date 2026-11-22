#!/bin/bash
cd "$(dirname "$0")"
.venv/bin/streamlit run app.py --server.headless true --server.port 8501 --server.fileWatcherType none
