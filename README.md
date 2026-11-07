# 🥁 OpenMoises — AI Drum Practice Studio

> **Disclaimer / 免责声明**: This project is an independent open-source personal/educational tool for drum practice and music analysis. It is **not affiliated with, endorsed by, or connected to Moises.ai (Moises Systems Inc.)**.  
> 本项目是一个独立的开源个人架子鼓练习与音频分析工具，**与 Moises.ai (Moises Systems Inc.) 没有任何关联、合作或官方背书关系**。

An interactive, local-first **AI Drum Practice Studio** powered by **Meta Demucs (`htdemucs`)** 4-stem source separation and a custom **multi-band transient & 16-step drum groove transcription engine** (`drum_engine.py`).

Built for drummers who want accurate **Beat 1 (downbeat) locking**, **dynamic tempo tracking** (for songs that speed up/slow down or have long drumless intros), **real-time 4-track stem mixing** (`Drums`, `Bass`, `Vocals`, `Other`), and **bar-accurate 16-step groove visualization**.

---

## 🔒 Strict Separation of Code & Audio Data / 代码与音频数据完全分离

To respect music copyrights and keep the repository lightweight, **this repository contains pure code only** (`index.html`, `server.py`, `drum_engine.py`). No audio files, separated stems, or pre-computed commercial song datasets are tracked in Git.

All user audio files, local song catalogs, and AI-separated stems live inside the local git-ignored `data/` directory (or a custom directory specified via the `DRUM_DATA_DIR` environment variable):

```text
openmoises/
├── index.html             # Frontend Web Audio 4-stem mixer, 60fps canvas timeline & 16-step sequencer
├── server.py              # Local HTTP Range streaming server & background Demucs job manager
├── drum_engine.py         # Demucs (htdemucs) 4-stem separator & multi-band Beat 1 / groove analyzer
├── requirements.txt       # Python dependencies
├── start_studio.command   # One-click macOS launcher
└── data/                  # 🔒 Local-only (git-ignored) — place your own practice songs here
    ├── catalog.json       # Optional metadata/practice tips for your local songs
    ├── songs/             # Drop your .m4a / .mp3 / .wav / .flac songs here (auto-discovered!)
    ├── stems_cache/       # Auto-generated 4-track separated stems (drums, bass, vocals, other)
    └── songs_analysis.js  # Auto-generated dynamic beat & 16-step groove cache
```

---

## ✨ Key Features / 核心功能

1. **🎛️ 4-Track AI Stem Separation (Demucs `htdemucs`)**
   - Separates any imported song (`.m4a`, `.mp3`, `.wav`, `.flac`) into 4 isolated tracks:
     - 🥁 **Drums (架子鼓轨)** — practice along with boosted drums (`130%`), or mute drums (`0%`) for drumless backing track mode.
     - 🎸 **Bass (贝斯轨)** — lock in your kick drum with the bassline.
     - 🎤 **Vocals (人声轨)** & 🎹 **Other (吉他/键盘伴奏轨)**.
   - Independent volume sliders (`0%–150%`), **Mute (`M`)**, **Solo (`S`)**, and one-click presets (*Original Full Mix*, *Drums Solo*, *Drumless Backing*, *Drums + Bass*).
   - Automatically uses **Apple Silicon GPU (`MPS`)** or CUDA when available.

2. **🎯 Dynamic Beat Tracking & True "Beat 1" (第一拍) Finder**
   - Tracks real-time tempo fluctuations across every bar rather than assuming a rigid constant BPM.
   - Snaps beat grid timestamps (`±40 ms`) to actual isolated drum transients (`0.0 ms` median alignment error).
   - Automatically detects **drumless intros & mid-song breakdowns** so the visual beat counter waits until the drums actually enter and gives a **4-beat pre-entry countdown** (`准备进鼓: 1 - 2 - 3 - 4`).
   - Uses a 4-phase physical scoring model (`Sub-Kick on Beats 1 & 3 + Snare Backbeat on Beats 2 & 4 + Onset Accent`) to lock accurately onto **Beat 1 (第一拍)**.

3. **🎼 Multi-Band 16-Step Drum Groove Transcription**
   - Decomposes the isolated drum stem into 3 physical frequency bands with group-delay compensation (`scipy.ndimage.maximum_filter1d`) and cross-band bleed suppression:
     - **Kick (`底鼓咚`, `35–110 Hz`)**
     - **Snare (`军鼓哒`, `150–1200 Hz`, with sub-kick transient bleed rejection)**
     - **Hi-Hat / Cymbal (`踩镲次`, `7000–16000 Hz`, with 16th-note decay-tail suppression & local offbeat floor adaptation)**
   - Extracts each song section's actual 16-step groove (`1 e & a 2 e & a 3 e & a 4 e & a`) and generates an intuitive Chinese oral counting guide (**口传心授**).

4. **⏱️ Interactive Practice Controls**
   - **Visual 4-Beat Counter** with high-contrast **Beat 1** flash and real-time local BPM readout.
   - **3-Lane Scrolling Drum Highway** (`Hat / Snare / Kick`) with **Beat 1 bar lines** and section cue markers.
   - **Dynamic WebAudio Click Track (节拍器)** synchronized to the song's exact dynamic beats (high click on Beat 1, low click on Beats 2/3/4).
   - **Section Jump & A-B Loop**: Click any detected song section (Intro, Drum Entry, Chorus, Breakdown) to jump or loop.
   - **Pitch-Preserved Speed Control**: Practice at `0.5x`–`1.25x` without shifting pitch.

---

## 🚀 Quick Start / 快速开始

### 1. Install Dependencies & Pre-Download Demucs AI Model / 安装依赖并下载 Demucs 模型

Requires **Python 3.10+** and `ffmpeg`.

The **Demucs `htdemucs` 4-stem AI model** (`adefossez/HTDemucs`, ~81 MB) is **not stored as a raw binary in Git** to keep the repository lightweight. Instead, `drum_engine.py` automatically downloads and caches the pretrained weights into `.cache/huggingface/` via `demucs.pretrained.get_model("htdemucs")`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pre-download and verify the Demucs htdemucs 4-stem AI model (~81 MB)
./.venv/bin/python drum_engine.py --download-model
```

*(Note: Double-clicking `start_studio.command` on macOS will automatically create `.venv`, install `requirements.txt`, and run `--download-model` on first launch!)*

### 2. Start the Local Studio Server / 启动本地服务

```bash
./.venv/bin/python server.py
```

Or double-click `start_studio.command` on macOS. Then open:

👉 **[http://127.0.0.1:8765](http://127.0.0.1:8765)**

### 3. Import or Add Your Own Songs / 导入或添加本地歌曲

- **Option A — Browser Upload**: Click **`📂 导入任意音乐 (自动分轨+找1拍)`** in the left sidebar to upload any `.m4a`, `.mp3`, `.wav`, or `.flac` file.
- **Option B — Local Folder Auto-Discovery**: Drop your audio files into `data/songs/` (optionally named like `01 [入门-92BPM] Artist - Title.m4a`) and refresh the page — `server.py` will automatically discover and list them.
- **Option C — CLI Separation & Analysis**:
  ```bash
  ./.venv/bin/python drum_engine.py --file "path/to/song.m4a" --bpm 80
  ```

---

## 📄 License

MIT License.
