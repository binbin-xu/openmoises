#!/bin/zsh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
  echo "📦 First-time setup: creating Python virtual environment (.venv)..."
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

echo "🤖 Checking Demucs AI model (htdemucs)..."
./.venv/bin/python drum_engine.py --download-model

if ! lsof -i :8765 >/dev/null 2>&1; then
  echo "🥁 Starting OpenMoises AI Drum Practice Studio (Demucs + Librosa)..."
  ./.venv/bin/python server.py &
  sleep 1.2
fi

open "http://127.0.0.1:8765"
wait
