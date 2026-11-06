import os
import re
import json
import uuid
import hashlib
import urllib.parse
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import drum_engine

PORT = int(os.environ.get("PORT", 8765))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = drum_engine.DATA_DIR
SONGS_DIR = drum_engine.SONGS_DIR
UPLOADS_DIR = os.path.join(drum_engine.CACHE_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()


def load_local_catalog():
    """
    Load local practice song catalog from data/catalog.json if present,
    and automatically discover any additional audio files placed in data/songs/.
    Keeps code 100% decoupled from copyrighted music assets.
    """
    catalog_path = os.path.join(DATA_DIR, "catalog.json")
    catalog = []
    seen_files = set()
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                raw_cat = json.load(f)
                for item in raw_cat:
                    fn = item.get("file", "")
                    if os.path.exists(os.path.join(SONGS_DIR, fn)) or os.path.exists(os.path.join(DATA_DIR, fn)) or os.path.exists(os.path.join(ROOT_DIR, fn)):
                        catalog.append(item)
                        seen_files.add(fn)
        except Exception:
            pass

    # Auto-discover any extra audio files in data/songs/
    if os.path.exists(SONGS_DIR):
        for fn in sorted(os.listdir(SONGS_DIR)):
            if fn in seen_files or fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac"):
                stem = os.path.splitext(fn)[0]
                m = re.match(r"^(\d+)\s*\[([^-]+)-(\d+)BPM\]\s*(.+)$", stem, re.I)
                if m:
                    num, level, bpm, title = m.group(1), m.group(2), int(m.group(3)), m.group(4).strip()
                else:
                    num = f"{len(catalog) + 1:02d}"
                    bpm_m = re.search(r"(\d{2,3})\s*BPM", stem, re.I)
                    bpm = int(bpm_m.group(1)) if bpm_m else 100
                    level = "自定义"
                    title = stem
                sid = re.sub(r"[^a-z0-9]+", "_", f"{num}_{title.lower()}").strip("_")[:32]
                catalog.append({
                    "id": sid,
                    "num": num,
                    "level": level,
                    "bpm": bpm,
                    "file": fn,
                    "title": title,
                    "tip": "本地曲库自动识别曲目 — 点击「AI 分离当前曲目四轨」可提取独立纯鼓轨与 16 分音符鼓谱。",
                })
                seen_files.add(fn)
    return catalog


def resolve_audio_file(file_name):
    """Resolve an audio filename across data/songs/, data/, and ROOT_DIR."""
    for base in (SONGS_DIR, DATA_DIR, ROOT_DIR):
        cand = os.path.join(base, file_name)
        if os.path.isfile(cand):
            return cand
    return None


def update_job(job_id, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def run_separation_job(job_id, audio_path, song_id, target_bpm=None, original_url=None):
    try:
        def cb(pct, msg):
            update_job(job_id, progress=pct, message=msg)

        drum_engine.separate_stems_demucs(audio_path, song_id, progress_cb=cb)
        update_job(job_id, progress=92, message="正在基于分离出的纯鼓音轨分析 Beat 1 强拍、动态速度与 16 分音符鼓谱...")
        analysis = drum_engine.analyze_song(audio_path, song_id, target_bpm=target_bpm, force_recompute=True)
        if original_url:
            analysis["audioUrl"] = original_url
        update_job(
            job_id,
            status="done",
            progress=100,
            message="✅ AI 四轨分离 (鼓/贝斯/人声/伴奏) 与鼓谱找拍分析完成！",
            result=analysis,
        )
    except Exception as e:
        update_job(job_id, status="error", message=f"分析失败: {e}")


class StudioHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

    def translate_path(self, path):
        """Map /stems_cache/* and song audio files to DATA_DIR while serving app code from ROOT_DIR."""
        parsed = urllib.parse.urlparse(path)
        rel = urllib.parse.unquote(parsed.path).lstrip("/")

        if rel.startswith("stems_cache/"):
            sub_rel = rel[len("stems_cache/"):]
            cand = os.path.join(drum_engine.CACHE_ROOT, sub_rel)
            if os.path.exists(cand):
                return cand

        if rel.startswith("songs/"):
            sub_rel = rel[len("songs/"):]
            cand = resolve_audio_file(sub_rel)
            if cand:
                return cand

        # Direct audio filename request (e.g. "/04 [入门-80BPM] Green Day - 21 Guns.m4a")
        if any(rel.lower().endswith(ext) for ext in (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac")):
            cand = resolve_audio_file(rel)
            if cand:
                return cand

        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-File-Name, X-Target-Bpm, Range")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            # Normal behavior when browser <audio> aborts a range stream after buffering
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/songs_analysis.js":
            catalog = load_local_catalog()
            js_parts = [f"window.LOCAL_SONG_CATALOG = {json.dumps(catalog, ensure_ascii=False)};\n"]
            analysis_js_path = os.path.join(DATA_DIR, "songs_analysis.js")
            if not os.path.exists(analysis_js_path):
                analysis_js_path = os.path.join(ROOT_DIR, "songs_analysis.js")
            if os.path.exists(analysis_js_path):
                with open(analysis_js_path, "r", encoding="utf-8") as f:
                    js_parts.append(f.read())
            else:
                js_parts.append("window.PRECOMPUTED_SONGS = window.PRECOMPUTED_SONGS || {};\n")

            # Also merge any individual stems_cache/<song_id>/analysis.json
            extra_analyses = {}
            if os.path.exists(drum_engine.CACHE_ROOT):
                for entry in os.listdir(drum_engine.CACHE_ROOT):
                    a_path = os.path.join(drum_engine.CACHE_ROOT, entry, "analysis.json")
                    if os.path.isfile(a_path):
                        try:
                            with open(a_path, "r", encoding="utf-8") as af:
                                extra_analyses[entry] = json.load(af)
                        except Exception:
                            pass
            if extra_analyses:
                js_parts.append(
                    f"\nwindow.PRECOMPUTED_SONGS = Object.assign(window.PRECOMPUTED_SONGS || {{}}, {json.dumps(extra_analyses, ensure_ascii=False)});\n"
                )

            payload = "".join(js_parts).encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if path == "/api/status":
            cached_stems = {}
            if os.path.exists(drum_engine.CACHE_ROOT):
                for entry in os.listdir(drum_engine.CACHE_ROOT):
                    sub = os.path.join(drum_engine.CACHE_ROOT, entry)
                    if os.path.isdir(sub) and all(
                        os.path.exists(os.path.join(sub, f"{s}.m4a"))
                        for s in ["drums", "bass", "other", "vocals"]
                    ):
                        cached_stems[entry] = {
                            s: f"stems_cache/{entry}/{s}.m4a"
                            for s in ["drums", "bass", "other", "vocals"]
                        }
            return self.send_json({
                "ok": True,
                "engine": "Demucs htdemucs (Apple Silicon MPS) + Librosa Multi-Band Drum Tracker",
                "dataDir": DATA_DIR,
                "catalog": load_local_catalog(),
                "cachedStems": cached_stems,
            })

        if path.startswith("/api/job/"):
            job_id = path.split("/api/job/")[1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                return self.send_json({"error": "Job not found"}, 404)
            return self.send_json(job)

        if path.startswith("/api/analysis/"):
            song_id = urllib.parse.unquote(path.split("/api/analysis/")[1])
            json_path = os.path.join(drum_engine.CACHE_ROOT, song_id, "analysis.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    return self.send_json(json.load(f))
            return self.send_json({"error": "Analysis not found"}, 404)

        # Support HTTP Range requests for smooth audio seeking in Safari/Chrome
        range_header = self.headers.get("Range")
        if range_header and any(path.lower().endswith(ext) for ext in (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac")):
            fs_path = self.translate_path(self.path)
            if os.path.isfile(fs_path):
                file_size = os.path.getsize(fs_path)
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    if start <= end:
                        length = end - start + 1
                        try:
                            self.send_response(206)
                            self.send_header("Content-Type", self.guess_type(fs_path))
                            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                            self.send_header("Content-Length", str(length))
                            self.end_headers()
                            with open(fs_path, "rb") as f:
                                f.seek(start)
                                remaining = length
                                while remaining > 0:
                                    chunk = f.read(min(65536, remaining))
                                    if not chunk:
                                        break
                                    self.wfile.write(chunk)
                                    remaining -= len(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        return

        try:
            return super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/separate_builtin":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            song_id = body["songId"]
            file_name = body["file"]
            bpm = body.get("bpm")
            audio_path = resolve_audio_file(file_name)
            if not audio_path:
                return self.send_json({"error": f"File not found: {file_name}"}, 404)

            job_id = str(uuid.uuid4())
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "jobId": job_id,
                    "status": "running",
                    "progress": 5,
                    "message": "已启动 AI 音轨分离任务...",
                }
            t = threading.Thread(
                target=run_separation_job,
                args=(job_id, audio_path, song_id, bpm, None),
                daemon=True,
            )
            t.start()
            return self.send_json({"jobId": job_id})

        if path == "/api/upload_and_analyze":
            length = int(self.headers.get("Content-Length", 0))
            raw_bytes = self.rfile.read(length)
            raw_name = urllib.parse.unquote(self.headers.get("X-File-Name", "uploaded_audio.m4a"))
            bpm_header = self.headers.get("X-Target-Bpm", "")
            target_bpm = float(bpm_header) if bpm_header and bpm_header.replace(".", "").isdigit() else None

            # Check if filename matches any song in the local catalog
            clean_lower = raw_name.lower()
            matched_local = None
            for item in load_local_catalog():
                title_part = item.get("title", "").split(" - ")[-1].lower()
                if (title_part and len(title_part) >= 3 and title_part in clean_lower) or clean_lower == item.get("file", "").lower():
                    resolved = resolve_audio_file(item["file"])
                    if resolved:
                        matched_local = (item["id"], resolved, item.get("bpm"), item["file"])
                        break

            if matched_local:
                song_id, audio_path, b_bpm, rel_url = matched_local
                target_bpm = target_bpm or b_bpm
            else:
                h = hashlib.md5(raw_bytes[:65536] + str(len(raw_bytes)).encode()).hexdigest()[:10]
                ext = os.path.splitext(raw_name)[1] or ".m4a"
                safe_ext = ext if ext.lower() in (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac") else ".m4a"
                song_id = f"custom_{h}"
                audio_path = os.path.join(UPLOADS_DIR, f"{song_id}{safe_ext}")
                with open(audio_path, "wb") as f:
                    f.write(raw_bytes)
                rel_url = f"stems_cache/uploads/{song_id}{safe_ext}"

            job_id = str(uuid.uuid4())
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "jobId": job_id,
                    "status": "running",
                    "progress": 5,
                    "message": f"正在使用 Demucs AI 分离《{raw_name}》的四条音轨...",
                }
            t = threading.Thread(
                target=run_separation_job,
                args=(job_id, audio_path, song_id, target_bpm, rel_url),
                daemon=True,
            )
            t.start()
            return self.send_json({"jobId": job_id, "songId": song_id})

        return self.send_json({"error": "Unknown endpoint"}, 404)


if __name__ == "__main__":
    ThreadingHTTPServer.allow_reuse_address = True
    print(f"🥁 OpenMoises AI Drum Practice Studio Server running at http://127.0.0.1:{PORT}")
    print(f"📂 Local Data Directory (git-ignored): {DATA_DIR}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), StudioHTTPRequestHandler)
    server.serve_forever()
