#!/bin/zsh
cd "/Users/binbinxu/Downloads/Drum_Practice_Songs"
if ! lsof -i :8765 >/dev/null 2>&1; then
  echo "🥁 正在启动 Rock Drum AI Practice Studio (Demucs + Librosa)..."
  /Users/binbinxu/Downloads/Drum_Practice_Songs/.venv/bin/python /Users/binbinxu/Downloads/Drum_Practice_Songs/server.py &
  sleep 1.2
fi
open "http://127.0.0.1:8765"
wait
