import os
import json
import hashlib
import subprocess
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import librosa

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DRUM_DATA_DIR", os.path.join(ROOT_DIR, "data"))
SONGS_DIR = os.path.join(DATA_DIR, "songs")
CACHE_ROOT = os.path.join(DATA_DIR, "stems_cache")
os.makedirs(SONGS_DIR, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".cache", "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT_DIR, ".cache", "torch"))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp")

FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"
if not os.path.exists(FFMPEG_BIN):
    FFMPEG_BIN = "ffmpeg"


def load_audio_mono(path, sr=44100):
    cmd = [FFMPEG_BIN, "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.float32)


def load_audio_stereo(path, sr=44100):
    cmd = [FFMPEG_BIN, "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def separate_stems_demucs(audio_path, song_id, progress_cb=None):
    """Run Demucs htdemucs 4-stem separation (drums, bass, other, vocals) and save as .m4a."""
    out_dir = os.path.join(CACHE_ROOT, song_id)
    os.makedirs(out_dir, exist_ok=True)
    expected = ["drums", "bass", "other", "vocals"]
    if all(os.path.exists(os.path.join(out_dir, f"{s}.m4a")) for s in expected):
        if progress_cb:
            progress_cb(100, "已命中本地 AI 分轨缓存")
        return {s: f"stems_cache/{song_id}/{s}.m4a" for s in expected}

    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    if progress_cb:
        progress_cb(10, "正在加载 Demucs (htdemucs) AI 模型...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = get_model("htdemucs").to(device).eval()
    sr = model.samplerate

    if progress_cb:
        progress_cb(20, "正在解码音频流...")
    wav = load_audio_stereo(audio_path, sr=sr)

    if progress_cb:
        progress_cb(35, f"正在使用 Apple Silicon ({device.upper()}) 分离 4 条音轨 (鼓/贝斯/人声/伴奏)...")

    with torch.no_grad():
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)
        sources = apply_model(
            model, wav_tensor, device=device, shifts=0, split=True, overlap=0.25, progress=False
        )[0].cpu().numpy()

    if progress_cb:
        progress_cb(85, "正在编码保存分离后的高音质音轨...")

    stems_map = {}
    for idx, name in enumerate(model.sources):
        stem_path = os.path.join(out_dir, f"{name}.m4a")
        stem_pcm = sources[idx].T.astype(np.float32).tobytes()
        p = subprocess.Popen(
            [
                FFMPEG_BIN, "-y", "-v", "quiet",
                "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                "-c:a", "aac", "-b:a", "160k", stem_path
            ],
            stdin=subprocess.PIPE
        )
        p.communicate(input=stem_pcm)
        stems_map[name] = f"stems_cache/{song_id}/{name}.m4a"

    if progress_cb:
        progress_cb(95, "音轨分离完成，正在生成鼓谱与节拍对齐数据...")
    return stems_map


def estimate_rock_bpm(drum_env, sr_env, hint_bpm=None):
    """Estimate BPM from drum onset envelope if no hint_bpm is given."""
    if hint_bpm and 50 <= hint_bpm <= 220:
        return float(hint_bpm)
    env = drum_env - np.mean(drum_env)
    if np.std(env) < 1e-6:
        return 100.0
    ac = np.correlate(env, env, mode="full")[len(env) - 1 : len(env) - 1 + int(sr_env * 2.0)]
    best_bpm = 100.0
    best_val = -1.0
    for bpm in np.linspace(68, 175, 215):
        lag = int(round((60.0 / bpm) * sr_env))
        if 1 <= lag < len(ac):
            val = ac[lag] + 0.5 * ac[min(len(ac) - 1, lag * 2)]
            if val > best_val:
                best_val = val
                best_bpm = bpm
    return float(best_bpm)


def analyze_song(audio_path, song_id, target_bpm=None, force_recompute=False):
    """
    Verified Multi-Band Drum & Beat Analyzer:
    - Uses isolated Demucs drums stem when available
    - Sub-band separation with group-delay compensation:
      * Kick fundamental: 35 - 110 Hz
      * Snare body & punch: 150 - 1200 Hz (with +/-25ms max-filtered Kick harmonic suppression)
      * Hi-Hat / Cymbal sizzle: 7000 - 16000 Hz (with 16th-note decay-tail suppression & local adaptive floor)
    - Verified Beat 1 downbeat phase scoring & dynamic beat tracking
    """
    out_dir = os.path.join(CACHE_ROOT, song_id)
    os.makedirs(out_dir, exist_ok=True)
    analysis_json_path = os.path.join(out_dir, "analysis.json")

    drums_stem_path = os.path.join(out_dir, "drums.m4a")
    has_stems = all(
        os.path.exists(os.path.join(out_dir, f"{s}.m4a"))
        for s in ["drums", "bass", "other", "vocals"]
    )

    if os.path.exists(analysis_json_path) and not force_recompute:
        try:
            with open(analysis_json_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("hasStems") == has_stems and cached.get("version") == 2:
                return cached
        except Exception:
            pass

    sr = 44100
    mix = load_audio_mono(audio_path, sr=sr)
    duration = float(len(mix)) / sr

    if os.path.exists(drums_stem_path):
        drums = load_audio_mono(drums_stem_path, sr=sr)
        if len(drums) > len(mix):
            drums = drums[:len(mix)]
        elif len(drums) < len(mix):
            drums = np.pad(drums, (0, len(mix) - len(drums)))
        source_mode = "demucs_drums"
    else:
        mix_22k = librosa.resample(mix, orig_sr=sr, target_sr=22050)
        _, perc_22k = librosa.effects.hpss(mix_22k, margin=(1.4, 3.0))
        drums = librosa.resample(perc_22k, orig_sr=22050, target_sr=sr)
        if len(drums) > len(mix):
            drums = drums[:len(mix)]
        elif len(drums) < len(mix):
            drums = np.pad(drums, (0, len(mix) - len(drums)))
        source_mode = "hpss_percussive"

    hop_200 = 220
    sr_200 = sr / hop_200

    def calc_band_env(sig, f0=None, f1=None):
        if f0 is not None and f1 is not None:
            sos = signal.butter(4, [f0, f1], btype="bandpass", fs=sr, output="sos")
            y_b = signal.sosfiltfilt(sos, sig)
        else:
            y_b = sig
        n = len(y_b) // hop_200
        frames = y_b[: n * hop_200].reshape(n, hop_200)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        diff = np.maximum(0, np.diff(rms, prepend=rms[0]))
        win = np.hanning(5)
        win /= win.sum()
        onset = np.convolve(diff, win, mode="same")
        return rms, onset

    # Physical drum frequency bands verified on isolated Demucs stems:
    # 1. Sub-Kick (35 - 110 Hz): pure bass drum fundamental
    k_rms, k_raw = calc_band_env(drums, 35, 110)
    # 2. Snare Body & Mid Punch (150 - 1200 Hz): snare fundamental + body (avoids crash sizzle >2kHz)
    s_rms, s_raw = calc_band_env(drums, 150, 1200)
    # 3. Crisp Hi-Hat & Cymbal Sizzle (7000 - 16000 Hz)
    h_rms, h_raw = calc_band_env(drums, 7000, 16000)
    d_rms, d_on = calc_band_env(drums)
    m_rms, m_on = calc_band_env(mix)

    def norm99(x):
        nz = x[x > 1e-5]
        p = np.percentile(nz, 99) if len(nz) > 0 else 1.0
        return np.clip(x / max(p, 1e-6), 0, 1.5)

    k_n = norm99(k_raw)
    sb_n = norm99(s_raw)
    h_n = norm99(h_raw)
    m_n = norm99(m_rms)

    # Group-delay-compensated cross-talk suppression (+/-25ms = 11 frames at 200Hz):
    # Low-frequency 35-110Hz Kick filter peaks ~10-15ms after the 150-1200Hz beater transient.
    k_max_env = ndimage.maximum_filter1d(k_n, size=11)
    s_clean = np.maximum(0, sb_n - 0.78 * k_max_env)
    s_max_env = ndimage.maximum_filter1d(s_clean, size=7)
    k_clean = np.maximum(0, k_n - 0.25 * s_max_env)

    # Detect active full-kit drum sections (where Kick or Snare hits occur)
    if source_mode == "demucs_drums":
        hit_mask = (k_clean > 0.36) | (s_clean > 0.40)
    else:
        # In HPSS fallback mode, require BOTH kick and snare activity in a local 3s window to avoid acoustic guitar false positives
        k_active = ndimage.maximum_filter1d(k_clean, size=int(2.5 * sr_200)) > 0.50
        s_active = ndimage.maximum_filter1d(s_clean, size=int(2.5 * sr_200)) > 0.50
        hit_mask = ((k_clean > 0.45) | (s_clean > 0.48)) & k_active & s_active

    hit_frames = np.where(hit_mask)[0]
    hit_times = hit_frames / sr_200

    raw_sections = []
    if len(hit_times) > 0:
        s_start = hit_times[0]
        prev_t = hit_times[0]
        count = 1
        for t in hit_times[1:]:
            if t - prev_t > 5.5:
                if count >= 6 and (prev_t - s_start) >= 2.5:
                    raw_sections.append((float(s_start), float(prev_t)))
                s_start = t
                count = 1
            else:
                count += 1
            prev_t = t
        if count >= 6 and (prev_t - s_start) >= 2.5:
            raw_sections.append((float(s_start), float(prev_t)))

    if not raw_sections:
        raw_sections = [(0.0, duration)]

    first_drum_sec = raw_sections[0][0]

    # Beat tracking with Librosa at 22050Hz
    sr_lib = 22050
    hop_lib = 256
    y_drums_22k = librosa.resample(drums, orig_sr=sr, target_sr=sr_lib)
    y_mix_22k = librosa.resample(mix, orig_sr=sr, target_sr=sr_lib)
    env_drums = librosa.onset.onset_strength(y=y_drums_22k, sr=sr_lib, hop_length=hop_lib)
    env_mix = librosa.onset.onset_strength(y=y_mix_22k, sr=sr_lib, hop_length=hop_lib)
    min_len = min(len(env_drums), len(env_mix))
    env_drums = env_drums[:min_len]
    env_mix = env_mix[:min_len]

    combined_env = (env_drums / (np.percentile(env_drums, 98) + 1e-6)) * 5.0 + (
        env_mix / (np.percentile(env_mix, 98) + 1e-6)
    ) * 0.45

    est_bpm = estimate_rock_bpm(k_clean + s_clean, sr_200, target_bpm)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=combined_env,
        sr=sr_lib,
        hop_length=hop_lib,
        start_bpm=est_bpm,
        tightness=85,
    )
    raw_beats = librosa.frames_to_time(beat_frames, sr=sr_lib, hop_length=hop_lib)

    if len(raw_beats) >= 4 and target_bpm:
        med_dur = float(np.median(np.diff(raw_beats)))
        tracked_bpm = 60.0 / max(med_dur, 1e-3)
        if tracked_bpm < target_bpm * 0.65:
            doubled = []
            for i in range(len(raw_beats) - 1):
                doubled.append(raw_beats[i])
                doubled.append(0.5 * (raw_beats[i] + raw_beats[i + 1]))
            doubled.append(raw_beats[-1])
            raw_beats = np.array(doubled)
        elif tracked_bpm > target_bpm * 1.55:
            even_b = raw_beats[0::2]
            odd_b = raw_beats[1::2]
            f_even = np.clip(np.round(even_b * sr_200).astype(int), 0, len(k_clean) - 1)
            f_odd = np.clip(np.round(odd_b * sr_200).astype(int), 0, len(k_clean) - 1)
            if np.mean(k_clean[f_odd] + s_clean[f_odd]) > np.mean(k_clean[f_even] + s_clean[f_even]):
                raw_beats = odd_b
            else:
                raw_beats = even_b

    active_raw_beats = raw_beats[raw_beats >= first_drum_sec - 0.15]
    if len(active_raw_beats) < 8:
        active_raw_beats = raw_beats

    # Snap each beat to nearest strong Kick or Snare transient within +/- 40ms
    snapped_beats = []
    for t in active_raw_beats:
        f = int(round(t * sr_200))
        w0, w1 = max(0, f - 8), min(len(k_n), f + 9)
        ks = k_clean[w0:w1] + s_clean[w0:w1] + 0.25 * h_n[w0:w1]
        if len(ks) > 0 and np.max(ks) > 0.35:
            best_f = w0 + int(np.argmax(ks))
            snapped_beats.append(float(best_f / sr_200))
        else:
            snapped_beats.append(float(t))
    snapped_beats = np.array(snapped_beats)

    # Determine the bar phase (0, 1, 2, 3) on active drum beats
    phase_scores = []
    for p in range(4):
        b1_k, b2_s, b3_k, b4_s = [], [], [], []
        b1_s, b2_k, b3_s, b4_k = [], [], [], []
        b1_hat = []
        for idx, t in enumerate(snapped_beats):
            if not any(sa - 0.2 <= t <= sb + 0.2 for sa, sb in raw_sections):
                continue
            f = int(round(t * sr_200))
            w0, w1 = max(0, f - 6), min(len(k_clean), f + 7)
            kv = float(np.max(k_clean[w0:w1])) if w1 > w0 else 0.0
            sv = float(np.max(s_clean[w0:w1])) if w1 > w0 else 0.0
            hv = float(np.max(h_n[w0:w1])) if w1 > w0 else 0.0
            role = (idx - p) % 4
            if role == 0:
                b1_k.append(kv); b1_s.append(sv); b1_hat.append(hv)
            elif role == 1:
                b2_s.append(sv); b2_k.append(kv)
            elif role == 2:
                b3_k.append(kv); b3_s.append(sv)
            elif role == 3:
                b4_s.append(sv); b4_k.append(kv)

        sc = (
            1.65 * np.mean(b1_k or [0])
            + 0.85 * np.mean(b3_k or [0])
            + 1.60 * np.mean(b2_s or [0])
            + 1.60 * np.mean(b4_s or [0])
            - 1.10 * np.mean(b1_s or [0])
            - 0.85 * np.mean(b3_s or [0])
            - 0.85 * np.mean(b2_k or [0])
            - 0.85 * np.mean(b4_k or [0])
            + 0.25 * np.mean(b1_hat or [0])
        )
        if p == 0 and len(snapped_beats) > 0:
            f0 = int(round(snapped_beats[0] * sr_200))
            w0, w1 = max(0, f0 - 6), min(len(k_clean), f0 + 7)
            if w1 > w0 and np.max(k_clean[w0:w1]) > 0.6:
                sc += 0.30
        phase_scores.append(float(sc))

    best_phase = int(np.argmax(phase_scores))
    first_beat1_idx = best_phase
    first_beat1_time = float(snapped_beats[first_beat1_idx]) if len(snapped_beats) > first_beat1_idx else 0.0

    if len(snapped_beats) >= 8:
        init_period = float(np.median(np.diff(snapped_beats[:min(16, len(snapped_beats))])))
    else:
        init_period = 60.0 / (target_bpm or 100.0)

    intro_beats = []
    t_cursor = float(snapped_beats[0])
    role_cursor = ((-best_phase) % 4)
    while t_cursor - init_period >= 0.05:
        t_cursor -= init_period
        role_cursor = (role_cursor - 1) % 4
        intro_beats.append((round(t_cursor, 4), role_cursor + 1))
    intro_beats.reverse()

    full_beats = []
    for t_val, b_in_bar in intro_beats:
        full_beats.append({"t": round(float(t_val), 4), "beat": int(b_in_bar), "inDrums": False})

    for idx, t_val in enumerate(snapped_beats):
        b_in_bar = ((idx - best_phase) % 4) + 1
        in_d = any(sa - 0.25 <= t_val <= sb + 0.25 for sa, sb in raw_sections)
        full_beats.append({"t": round(float(t_val), 4), "beat": int(b_in_bar), "inDrums": bool(in_d)})

    if len(full_beats) >= 2:
        tail_period = init_period
        t_cursor = full_beats[-1]["t"]
        role_cursor = full_beats[-1]["beat"] - 1
        while t_cursor + tail_period < duration - 0.1:
            t_cursor += tail_period
            role_cursor = (role_cursor + 1) % 4
            full_beats.append({"t": round(float(t_cursor), 4), "beat": int(role_cursor + 1), "inDrums": False})

    bar_num = 1
    for i, b in enumerate(full_beats):
        if i > 0 and b["beat"] == 1:
            bar_num += 1
        b["bar"] = bar_num
        if i < len(full_beats) - 1:
            dt = max(0.2, full_beats[i + 1]["t"] - b["t"])
            b["bpm"] = round(60.0 / dt, 1)
        elif i > 0:
            b["bpm"] = full_beats[i - 1]["bpm"]
        else:
            b["bpm"] = round(60.0 / init_period, 1)

    # Build verified per-bar 16-step drum groove transcription
    bars_data = []
    bar_groups = {}
    for b in full_beats:
        bar_groups.setdefault(b["bar"], []).append(b)

    step_names = ["1", "e", "&", "a", "2", "e", "&", "a", "3", "e", "&", "a", "4", "e", "&", "a"]

    for b_num in sorted(bar_groups.keys()):
        b_list = bar_groups[b_num]
        if len(b_list) < 4 or b_list[0]["beat"] != 1:
            continue
        t_start = b_list[0]["t"]
        next_bar = bar_groups.get(b_num + 1)
        if next_bar and next_bar[0]["beat"] == 1:
            t_end = next_bar[0]["t"]
        else:
            t_end = b_list[-1]["t"] + (b_list[-1]["t"] - b_list[0]["t"]) / 3.0

        kick_steps = [0] * 16
        snare_steps = [0] * 16
        hat_steps = [0] * 16
        raw_h_bar = np.zeros(16)

        beat_times_4 = [b_list[k]["t"] for k in range(4)] + [t_end]
        for q in range(4):
            q_start = beat_times_4[q]
            q_end = beat_times_4[q + 1]
            for sub in range(4):
                step_idx = q * 4 + sub
                st_time = q_start + (sub / 4.0) * (q_end - q_start)
                f = int(round(st_time * sr_200))
                w0, w1 = max(0, f - 5), min(len(k_clean), f + 6)
                kv = float(np.max(k_clean[w0:w1])) if w1 > w0 else 0.0
                sv = float(np.max(s_clean[w0:w1])) if w1 > w0 else 0.0
                hv = float(np.max(h_n[w0:w1])) if w1 > w0 else 0.0
                raw_h_bar[step_idx] = float(np.max(h_raw[w0:w1])) if w1 > w0 else 0.0

                if kv > 0.65:
                    kick_steps[step_idx] = 2
                elif kv > 0.38:
                    kick_steps[step_idx] = 1

                if sv > 0.60:
                    snare_steps[step_idx] = 2
                elif sv > 0.42:
                    snare_steps[step_idx] = 1

                if hv > 0.55:
                    hat_steps[step_idx] = 2
                elif hv > 0.22:
                    hat_steps[step_idx] = 1

        # Suppress acoustic decay tail from previous 16th-note step (e.g. snare/crash ring into 'e')
        net_h_bar = np.zeros(16)
        for st in range(16):
            prev_h = raw_h_bar[(st - 1) % 16]
            net_h_bar[st] = max(0.0, raw_h_bar[st] - 0.15 * prev_h)
            if st % 2 == 1 and hat_steps[st] == 1 and raw_h_bar[st] < 0.30 * prev_h:
                hat_steps[st] = 0

        # Local adaptive Hi-Hat detection for closed 8th-note hi-hats in quiet verses
        offbeat_floor = float(np.median(net_h_bar[1::2]))
        if sum(kick_steps) + sum(snare_steps) > 0:
            for st in range(0, 16, 2):
                if hat_steps[st] == 0 and net_h_bar[st] > max(0.0006, offbeat_floor * 2.2):
                    hat_steps[st] = 1

        ks_hits = sum(1 for x in kick_steps + snare_steps if x > 0)
        h_hits = sum(1 for x in hat_steps if x > 0)
        snare_offbeat_hits = sum(1 for idx_s in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15] if snare_steps[idx_s] > 0)
        in_drums = any(b["inDrums"] for b in b_list)

        if (not in_drums and h_hits < 2) or (ks_hits == 0 and h_hits < 2):
            bar_type = "silent"
        elif snare_offbeat_hits >= 1:
            bar_type = "fill"
        else:
            bar_type = "groove"

        oral_beats = []
        for q in range(4):
            syllables = []
            for sub in range(4):
                si = q * 4 + sub
                k_on_s = kick_steps[si] > 0
                s_on_s = snare_steps[si] > 0
                h_on_s = hat_steps[si] > 0
                lbl = step_names[si]
                if k_on_s and s_on_s:
                    syllables.append(f"{lbl}:咚哒")
                elif s_on_s:
                    syllables.append(f"{lbl}:哒")
                elif k_on_s:
                    syllables.append(f"{lbl}:咚")
                elif h_on_s and sub in (0, 2):
                    syllables.append(f"{lbl}:次")
            oral_beats.append(" ".join(syllables) if syllables else f"{q+1}:休")

        bars_data.append({
            "bar": b_num,
            "t0": round(t_start, 3),
            "t1": round(t_end, 3),
            "bpm": b_list[0]["bpm"],
            "type": bar_type,
            "k": kick_steps,
            "s": snare_steps,
            "h": hat_steps,
            "oral": " ｜ ".join(oral_beats),
        })

    # Mark the bar immediately preceding each active drum section as "count_in"
    for i, br in enumerate(bars_data):
        if br["type"] in ("groove", "fill") and sum(br["k"]) + sum(br["s"]) > 0:
            if i > 0 and bars_data[i - 1]["type"] == "silent":
                bars_data[i - 1]["type"] = "count_in"
                bars_data[i - 1]["oral"] = "🔔 预备倒数：1 - 2 - 3 - 4 (下一小节第 1 拍进鼓！)"

    # Build Smart Practice Cue Markers
    cues = [{"label": "🎵 曲目开头", "time": 0.0, "type": "intro"}]
    for s_idx, (sa, sb) in enumerate(raw_sections):
        cand_bars = [br for br in bars_data if abs(br["t0"] - sa) < 2.5 and sum(br["k"]) + sum(br["s"]) > 0]
        entry_bar = cand_bars[0] if cand_bars else None
        entry_t = entry_bar["t0"] if entry_bar else round(sa, 2)
        entry_bar_num = entry_bar["bar"] if entry_bar else 1

        if s_idx == 0 and entry_t > 4.0:
            prev_bars = [br for br in bars_data if br["bar"] == entry_bar_num - 1]
            cue_t = prev_bars[0]["t0"] if prev_bars else max(0.0, entry_t - 4 * init_period)
            cues.append({
                "label": f"⏳ 进鼓预备倒数 (Bar {max(1, entry_bar_num-1)})",
                "time": round(cue_t, 2),
                "type": "count_in",
            })
            cues.append({
                "label": f"🥁 首次进鼓 Beat 1 (Bar {entry_bar_num} · {entry_t:.1f}s)",
                "time": round(entry_t, 2),
                "type": "drum_entry",
            })
        elif s_idx == 0:
            cues.append({
                "label": f"🥁 首拍进鼓 Beat 1 (Bar {entry_bar_num})",
                "time": round(entry_t, 2),
                "type": "drum_entry",
            })
        else:
            prev_bars = [br for br in bars_data if br["bar"] == entry_bar_num - 1]
            if prev_bars:
                cues.append({
                    "label": f"⏳ 间奏后预备拍 (Bar {entry_bar_num-1})",
                    "time": round(prev_bars[0]["t0"], 2),
                    "type": "count_in",
                })
            cues.append({
                "label": f"🚀 间奏再进鼓 Beat 1 (Bar {entry_bar_num})",
                "time": round(entry_t, 2),
                "type": "re_entry",
            })

    # Detect Chorus / High-Energy explosion bar (where both drum RMS and backbeat snare return/jump)
    if len(bars_data) > 8:
        for i in range(4, len(bars_data) - 2):
            br = bars_data[i]
            # Require chorus bar to have standard backbeat snare (s[4]>0 and s[12]>0) so we skip pre-chorus tom build bars!
            if br["t0"] > first_drum_sec + 12.0 and br["s"][4] > 0 and br["s"][12] > 0:
                f_curr = int(round(br["t0"] * sr_200))
                f_prev = int(round(bars_data[max(0, i - 4)]["t0"] * sr_200))
                rms_curr = float(np.mean(d_rms[f_curr : min(len(d_rms), f_curr + int(6 * sr_200))]))
                rms_prev = float(np.mean(d_rms[f_prev : min(len(d_rms), f_prev + int(6 * sr_200))]))
                if rms_curr > rms_prev * 1.45 and rms_curr > 0.07:
                    if all(abs(c["time"] - br["t0"]) > 10.0 for c in cues):
                        cues.append({
                            "label": f"💥 副歌爆发 Beat 1 (Bar {br['bar']})",
                            "time": round(br["t0"], 2),
                            "type": "chorus",
                        })
                        break

    hop_50 = 4
    def ds50(arr):
        n = len(arr) // hop_50
        reshaped = arr[: n * hop_50].reshape(n, hop_50)
        return [round(float(x), 3) for x in np.max(reshaped, axis=1)]

    kick_50 = ds50(k_clean)
    snare_50 = ds50(s_clean)
    hat_50 = ds50(h_n)
    mix_50 = ds50(m_n)

    groove_bars = [b for b in bars_data if b["type"] == "groove"]
    sig_bar = groove_bars[min(2, len(groove_bars) - 1)] if groove_bars else (bars_data[0] if bars_data else None)

    active_bpms = [b["bpm"] for b in full_beats if b["inDrums"]]
    median_bpm = round(float(np.median(active_bpms)) if active_bpms else est_bpm, 1)
    min_bpm = round(float(np.percentile(active_bpms, 10)) if len(active_bpms) > 4 else median_bpm, 1)
    max_bpm = round(float(np.percentile(active_bpms, 90)) if len(active_bpms) > 4 else median_bpm, 1)

    result = {
        "version": 2,
        "songId": song_id,
        "duration": round(duration, 3),
        "sourceMode": source_mode,
        "hasStems": has_stems,
        "stems": {
            s: f"stems_cache/{song_id}/{s}.m4a"
            for s in ["drums", "bass", "other", "vocals"]
        } if has_stems else None,
        "medianBpm": median_bpm,
        "bpmRange": [min_bpm, max_bpm],
        "firstDrumSec": round(first_drum_sec, 2),
        "firstBeat1Sec": round(first_beat1_time, 3),
        "cues": sorted(cues, key=lambda c: c["time"]),
        "signatureGroove": sig_bar,
        "envRate": round(sr_200 / hop_50, 4),
        "mixPeaks": mix_50,
        "kickEnv": kick_50,
        "snareEnv": snare_50,
        "hatEnv": hat_50,
        "beats": full_beats,
        "bars": bars_data,
    }

    with open(analysis_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    return result


def ensure_demucs_model():
    """
    Verify that the pretrained Demucs htdemucs model (adefossez/HTDemucs, ~81 MB)
    is present in local .cache/huggingface, downloading it automatically if missing.
    """
    import torch
    from demucs.pretrained import get_model

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🤖 Loading / verifying Demucs 'htdemucs' pretrained model (device={device})...")
    model = get_model("htdemucs").to(device).eval()
    print(f"✅ Demucs 'htdemucs' 4-stem model ready (sources: {model.sources}, samplerate: {model.samplerate} Hz)")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenMoises AI Drum Stem Separator & Beat/Groove Analyzer")
    parser.add_argument("--download-model", action="store_true", help="Download and verify the Demucs htdemucs model weights")
    parser.add_argument("--file", type=str, help="Path to an audio file (.m4a/.mp3/.wav/.flac) to separate and analyze")
    parser.add_argument("--song-id", type=str, default=None, help="Optional song ID for caching under data/stems_cache/<song_id>")
    parser.add_argument("--bpm", type=float, default=None, help="Optional target BPM hint")
    args = parser.parse_args()

    if args.download_model:
        ensure_demucs_model()
    elif args.file:
        audio_path = os.path.abspath(args.file)
        sid = args.song_id or os.path.splitext(os.path.basename(audio_path))[0].lower().replace(" ", "_")[:32]
        print(f"🎧 Step 1/2: Separating 4 stems with Demucs htdemucs for {os.path.basename(audio_path)}...")
        separate_stems_demucs(audio_path, sid, progress_cb=lambda p, m: print(f"  [{p:3d}%] {m}"))
        print("🥁 Step 2/2: Running multi-band Beat 1 & 16-step groove transcription...")
        res = analyze_song(audio_path, sid, target_bpm=args.bpm, force_recompute=True)
        print(f"✅ Done! Median BPM: {res['medianBpm']}, First Drum Entry: {res['firstDrumSec']}s, First Beat 1: {res['firstBeat1Sec']}s")
    else:
        parser.print_help()

