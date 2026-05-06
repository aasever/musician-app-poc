"""
Audio analysis engine: pitch detection, chord identification,
key detection, overtone analysis, and frequency-band source separation.
"""

import io
import tempfile
import os
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d


# ── Music theory constants ──────────────────────────────────────────────────

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ENHARMONIC  = {
    'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'
}

# Krumhansl–Schmuckler key-profiles
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Chord templates (1 = chord tone present)
CHORD_TYPES = {
    'maj':   [1,0,0,0,1,0,0,1,0,0,0,0],
    'min':   [1,0,0,1,0,0,0,1,0,0,0,0],
    '7':     [1,0,0,0,1,0,0,1,0,0,1,0],
    'maj7':  [1,0,0,0,1,0,0,1,0,0,0,1],
    'min7':  [1,0,0,1,0,0,0,1,0,0,1,0],
    'dim':   [1,0,0,1,0,0,1,0,0,0,0,0],
    'aug':   [1,0,0,0,1,0,0,0,1,0,0,0],
    'sus4':  [1,0,0,0,0,1,0,1,0,0,0,0],
    'sus2':  [1,0,1,0,0,0,0,1,0,0,0,0],
    'dim7':  [1,0,0,1,0,0,1,0,0,1,0,0],
    'add9':  [1,0,1,0,1,0,0,1,0,0,0,0],
}
CHORD_TEMPLATES = {
    f"{NOTE_NAMES[root]}{qtype}": (
        np.roll(np.array(template, dtype=float), root)
    )
    for qtype, template in CHORD_TYPES.items()
    for root in range(12)
}

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class NoteEvent:
    note:       str          # e.g. "A4"
    pitch_class: str         # e.g. "A"
    octave:     int
    midi:       int
    start_beat: float
    duration_beats: float
    frequency:  float
    confidence: float

@dataclass
class ChordEvent:
    chord:      str          # e.g. "Cmaj"
    notes:      list         # constituent pitch-classes
    start_beat: float
    duration_beats: float
    confidence: float

@dataclass
class OvertoneInfo:
    fundamental_hz:   float
    fundamental_note: str
    harmonics: list          # [{n, hz, note, relative_amplitude}]

@dataclass
class TrackAnalysis:
    label:      str          # "full", "bass", "mid", "treble"
    notes:      list = field(default_factory=list)   # [NoteEvent]
    chords:     list = field(default_factory=list)   # [ChordEvent]
    tempo_bpm:  float = 0.0
    key:        Optional[str] = None
    key_confidence: float = 0.0
    overtones:  list = field(default_factory=list)   # [OvertoneInfo]
    duration_sec: float = 0.0
    waveform_samples: list = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hz_to_note_event(hz: float, time_sec: float, duration: float,
                      confidence: float, bpm: float) -> Optional[NoteEvent]:
    if hz <= 0 or not np.isfinite(hz):
        return None
    midi_f = librosa.hz_to_midi(hz)
    midi   = int(round(midi_f))
    pc     = NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    beat   = (time_sec * bpm) / 60.0
    dur_b  = (duration * bpm) / 60.0
    return NoteEvent(
        note=f"{pc}{octave}",
        pitch_class=pc,
        octave=octave,
        midi=midi,
        start_beat=round(beat, 3),
        duration_beats=round(dur_b, 3),
        frequency=round(float(hz), 2),
        confidence=round(float(confidence), 3),
    )


def _bandpass(y: np.ndarray, sr: int,
              low_hz: float, high_hz: float) -> np.ndarray:
    nyq = sr / 2.0
    low  = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.9999)
    if low >= high:
        return y
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, y)


def _lowpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    nyq = sr / 2.0
    b, a = butter(4, min(cutoff_hz / nyq, 0.9999), btype='low')
    return filtfilt(b, a, y)


def _highpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    nyq = sr / 2.0
    b, a = butter(4, max(cutoff_hz / nyq, 1e-4), btype='high')
    return filtfilt(b, a, y)


def _downsample_waveform(y: np.ndarray, n_points: int = 800) -> list:
    if len(y) == 0:
        return []
    if len(y) <= n_points:
        return [round(float(v), 4) for v in y]
    step = len(y) / n_points
    idx  = [int(i * step) for i in range(n_points)]
    return [round(float(y[i]), 4) for i in idx]


# ── Key detection ────────────────────────────────────────────────────────────

def detect_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Returns (key_note, mode, correlation_score)."""
    best_r, best_note, best_mode = -np.inf, 'C', 'major'
    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        r_maj = float(np.corrcoef(rotated, KS_MAJOR)[0, 1])
        r_min = float(np.corrcoef(rotated, KS_MINOR)[0, 1])
        if r_maj > best_r:
            best_r, best_note, best_mode = r_maj, NOTE_NAMES[i], 'major'
        if r_min > best_r:
            best_r, best_note, best_mode = r_min, NOTE_NAMES[i], 'minor'
    return best_note, best_mode, round(best_r, 3)


# ── Chord detection ──────────────────────────────────────────────────────────

def _chroma_to_chord(chroma_vec: np.ndarray,
                     threshold: float = 0.35) -> Optional[ChordEvent]:
    """Match a single chroma vector to the best chord template."""
    norm = np.linalg.norm(chroma_vec)
    if norm < 1e-6:
        return None
    chroma_n = chroma_vec / norm

    best_name, best_score = None, -np.inf
    for name, tmpl in CHORD_TEMPLATES.items():
        score = float(np.dot(chroma_n, tmpl / (np.linalg.norm(tmpl) + 1e-9)))
        if score > best_score:
            best_score, best_name = score, name

    if best_score < threshold or best_name is None:
        return None

    # active pitch classes
    active = [NOTE_NAMES[i] for i, v in enumerate(chroma_vec)
              if v > 0.25 * chroma_vec.max()]
    return ChordEvent(
        chord=best_name,
        notes=active,
        start_beat=0.0,
        duration_beats=0.0,
        confidence=round(best_score, 3),
    )


def detect_chords(y: np.ndarray, sr: int, bpm: float,
                  hop_length: int = 4096) -> list:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times  = librosa.frames_to_time(np.arange(chroma.shape[1]),
                                    sr=sr, hop_length=hop_length)
    chords = []
    prev   = None
    chunk  = 4          # smooth over N frames before matching

    for start in range(0, chroma.shape[1], chunk):
        block  = chroma[:, start:start + chunk].mean(axis=1)
        event  = _chroma_to_chord(block)
        if event is None:
            prev = None
            continue
        beat = (times[start] * bpm) / 60.0
        dur  = (min(chunk, chroma.shape[1] - start)
                * (hop_length / sr) * bpm / 60.0)
        if prev and prev.chord == event.chord:
            prev.duration_beats += round(dur, 3)
            continue
        event.start_beat     = round(beat, 3)
        event.duration_beats = round(dur, 3)
        chords.append(event)
        prev = event

    return chords


# ── Note / pitch detection ───────────────────────────────────────────────────

def detect_notes(y: np.ndarray, sr: int, bpm: float,
                 fmin_hz: float = 40.0,
                 fmax_hz: float = 4200.0) -> list:
    """
    Detect monophonic or lightly polyphonic notes using pyin on
    the harmonic component, then merge consecutive identical pitches.
    """
    # Separate harmonic component for cleaner pitch tracking
    y_harm, _ = librosa.effects.hpss(y, margin=4)

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_harm,
            fmin=max(fmin_hz, librosa.note_to_hz('C1')),
            fmax=min(fmax_hz, librosa.note_to_hz('C8')),
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
    except Exception:
        return []

    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
    frame_dur = float(times[1] - times[0]) if len(times) > 1 else 0.023

    notes = []
    i = 0
    while i < len(f0):
        if not voiced_flag[i] or f0[i] is None or not np.isfinite(f0[i]):
            i += 1
            continue
        midi_now = int(round(librosa.hz_to_midi(f0[i])))
        j = i + 1
        conf_sum = float(voiced_probs[i])
        while (j < len(f0) and voiced_flag[j]
               and f0[j] is not None and np.isfinite(f0[j])
               and abs(int(round(librosa.hz_to_midi(f0[j]))) - midi_now) <= 1):
            conf_sum += float(voiced_probs[j])
            j += 1
        duration = (j - i) * frame_dur
        if duration < 0.04:       # skip very short blips
            i = j
            continue
        avg_hz   = float(np.nanmean(f0[i:j]))
        avg_conf = conf_sum / (j - i)
        ev = _hz_to_note_event(avg_hz, float(times[i]), duration, avg_conf, bpm)
        if ev:
            notes.append(ev)
        i = j

    return notes


# ── Overtone analysis ────────────────────────────────────────────────────────

def analyze_overtones(y: np.ndarray, sr: int,
                      n_harmonics: int = 8,
                      max_fundamentals: int = 5) -> list:
    """
    Find the strongest spectral peaks and label their harmonic series.
    """
    # Use a long FFT for frequency resolution
    n_fft = min(32768, len(y))
    segment = y[:n_fft] if len(y) >= n_fft else np.pad(y, (0, n_fft - len(y)))
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(n_fft), n=n_fft))
    freqs    = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Only look at musical range
    mask  = (freqs >= 40) & (freqs <= 5000)
    s_sub = spectrum[mask]
    f_sub = freqs[mask]

    if len(s_sub) == 0:
        return []

    # Find peaks
    min_dist = max(1, int(20 / (freqs[1] - freqs[0])))  # 20 Hz minimum separation
    peaks, props = find_peaks(s_sub, height=s_sub.max() * 0.03,
                              distance=min_dist)
    if len(peaks) == 0:
        return []

    peak_freqs = f_sub[peaks]
    peak_amps  = s_sub[peaks]
    order      = np.argsort(peak_amps)[::-1]

    infos = []
    used  = set()
    for idx in order:
        if len(infos) >= max_fundamentals:
            break
        fund_hz = float(peak_freqs[idx])
        # Skip if this freq is already a harmonic of a found fundamental
        if any(abs(fund_hz / f - round(fund_hz / f)) < 0.05
               for f in used if f > 0):
            continue
        used.add(fund_hz)

        midi_f = librosa.hz_to_midi(fund_hz)
        midi_i = int(round(midi_f))
        note   = f"{NOTE_NAMES[midi_i % 12]}{(midi_i // 12) - 1}"

        harmonics = []
        for n in range(1, n_harmonics + 1):
            h_hz     = fund_hz * n
            if h_hz > sr / 2:
                break
            # find nearest spectrum bin
            bin_idx  = int(round(h_hz * n_fft / sr))
            bin_idx  = min(bin_idx, len(spectrum) - 1)
            h_amp    = float(spectrum[bin_idx]) / (float(peak_amps[idx]) + 1e-9)
            h_midi   = int(round(librosa.hz_to_midi(h_hz))) if h_hz >= 16 else 0
            h_note   = (f"{NOTE_NAMES[h_midi % 12]}{(h_midi // 12) - 1}"
                        if h_midi > 0 else "?")
            harmonics.append({
                "n":                 n,
                "hz":                round(h_hz, 1),
                "note":              h_note,
                "relative_amplitude": round(min(h_amp, 1.0), 3),
            })

        infos.append(OvertoneInfo(
            fundamental_hz=round(fund_hz, 1),
            fundamental_note=note,
            harmonics=harmonics,
        ))

    return infos


# ── Source separation (frequency-band) ──────────────────────────────────────

def separate_bands(y: np.ndarray, sr: int) -> dict:
    """
    Split audio into three functional bands:
      bass   : 20 – 300 Hz  (bass instruments, low fundamentals)
      mid    : 200 – 2000 Hz (chord instruments, voices)
      treble : 1500 – 20000 Hz (lead, melody, highs)
    Bands overlap slightly so no important content is clipped.
    """
    return {
        "bass":   _bandpass(y, sr, 20,   320),
        "mid":    _bandpass(y, sr, 180, 2200),
        "treble": _highpass(y, sr, 1400),
    }


# ── Main entry point ─────────────────────────────────────────────────────────

def analyze_audio(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    """
    Full analysis pipeline. Returns a dict ready for JSON serialisation.
    """
    # ── Load audio ──────────────────────────────────────────────────────────
    suffix = os.path.splitext(filename)[-1].lower() or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=22050, mono=True, duration=90)
    except Exception as exc:
        return {"error": f"Could not decode audio: {exc}"}
    finally:
        os.unlink(tmp_path)

    if len(y) < sr * 0.1:
        return {"error": "Recording too short (< 0.1 s)"}

    duration = len(y) / sr

    # ── Tempo ────────────────────────────────────────────────────────────────
    try:
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo_arr)[0])
    except Exception:
        bpm = 0.0
    bpm = round(bpm if bpm > 0 else 120.0, 1)

    # ── Global chroma & key ──────────────────────────────────────────────────
    chroma  = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
    chroma_mean = chroma.mean(axis=1)
    key_note, key_mode, key_conf = detect_key(chroma_mean)
    key_str = f"{key_note} {key_mode}"

    # ── Per-band analysis ────────────────────────────────────────────────────
    bands    = separate_bands(y, sr)
    tracks   = {}

    band_cfg = {
        "full":   {"y": y,             "fmin": 40,  "fmax": 4200, "label": "Full Mix"},
        "bass":   {"y": bands["bass"], "fmin": 20,  "fmax": 350,  "label": "Bass"},
        "mid":    {"y": bands["mid"],  "fmin": 180, "fmax": 2200, "label": "Mid / Chords"},
        "treble": {"y": bands["treble"],"fmin":1400,"fmax": 8000, "label": "Lead / Melody"},
    }

    for band_id, cfg in band_cfg.items():
        yb    = cfg["y"]
        label = cfg["label"]

        # Skip silent bands
        rms = float(np.sqrt(np.mean(yb ** 2)))
        if rms < 1e-5:
            tracks[band_id] = {
                "label": label, "notes": [], "chords": [],
                "key": None, "key_confidence": 0.0,
                "tempo_bpm": bpm, "duration_sec": round(duration, 2),
                "waveform_samples": [],
                "overtones": [],
            }
            continue

        notes  = detect_notes(yb, sr, bpm, cfg["fmin"], cfg["fmax"])
        chords = detect_chords(yb, sr, bpm)

        # Per-band key
        chb      = librosa.feature.chroma_cqt(y=yb, sr=sr, hop_length=2048)
        kn, km, kc = detect_key(chb.mean(axis=1))

        overtones = []
        if band_id == "full":
            overtones = analyze_overtones(y, sr)

        def _nt(ev): return vars(ev)
        def _ch(ev): return vars(ev)
        def _ov(ev): return {
            "fundamental_hz":   ev.fundamental_hz,
            "fundamental_note": ev.fundamental_note,
            "harmonics":        ev.harmonics,
        }

        tracks[band_id] = {
            "label":          label,
            "notes":          [_nt(n) for n in notes],
            "chords":         [_ch(c) for c in chords],
            "key":            f"{kn} {km}",
            "key_confidence": kc,
            "tempo_bpm":      bpm,
            "duration_sec":   round(duration, 2),
            "waveform_samples": _downsample_waveform(yb),
            "overtones":      [_ov(o) for o in overtones],
        }

    return {
        "duration_sec":    round(duration, 2),
        "tempo_bpm":       bpm,
        "key":             key_str,
        "key_confidence":  round(key_conf, 3),
        "tracks":          tracks,
        "sample_rate":     sr,
        "waveform_samples": _downsample_waveform(y),
    }
