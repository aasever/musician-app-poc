"""
High-accuracy monophonic note detection.

Pipeline:
  1. Load at 44100 Hz (better high-freq resolution).
  2. Harmonic/percussive separation (HPSS) to isolate tonal content.
  3. pyin pitch tracking with a large frame window.
  4. Strict voiced-probability threshold (≥ 0.75).
  5. MIDI-space median filtering per voiced segment → removes single-frame
     octave flips and jitter without blurring real pitch changes.
  6. Octave-consistency correction → fixes the rare case where a brief
     sub-harmonic capture breaks an otherwise-stable note.
  7. Note merging: consecutive frames within 1.5 semitones are the same note.
  8. Cents-deviation annotation → how sharp/flat vs equal temperament.
"""

import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import librosa
from scipy.ndimage import median_filter, label


# ── Constants ────────────────────────────────────────────────────────────────

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# pyin tuning
SR            = 44100
FRAME_LENGTH  = 4096          # ~93 ms window → good freq resolution
HOP_LENGTH    = 512           # ~11.6 ms hop → smooth time tracking
VOICED_THRESH = 0.75          # minimum voiced probability to accept a frame
MIN_NOTE_DUR  = 0.08          # seconds – discard blips shorter than this
MERGE_CENTS   = 70            # merge consecutive frames within 0.7 semitones

# Pitch limits
FMIN = librosa.note_to_hz('C2')   # 65.4 Hz
FMAX = librosa.note_to_hz('C7')   # 2093 Hz

WAVEFORM_POINTS = 1000


# ── Data class ───────────────────────────────────────────────────────────────

@dataclass
class NoteEvent:
    note:          str    # e.g. "A4"
    pitch_class:   str    # e.g. "A"
    octave:        int
    midi:          int
    frequency_hz:  float  # mean detected Hz across the note
    cents_offset:  float  # how sharp(+) / flat(−) vs equal temperament
    start_sec:     float
    end_sec:       float
    duration_sec:  float
    start_beat:    float
    duration_beats: float
    confidence:    float  # mean voiced_prob for the note's frames


# ── Helpers ──────────────────────────────────────────────────────────────────

def _midi_to_note(midi: int) -> tuple[str, str, int]:
    pc  = NOTE_NAMES[midi % 12]
    oct = (midi // 12) - 1
    return f"{pc}{oct}", pc, oct


def _cents_offset(hz: float, midi: int) -> float:
    """Deviation in cents from equal-temperament pitch for this MIDI note."""
    ref_hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    return round(1200.0 * np.log2(hz / ref_hz), 1)


def _downsample(y: np.ndarray, n: int = WAVEFORM_POINTS) -> list:
    if len(y) == 0:
        return []
    if len(y) <= n:
        return [round(float(v), 4) for v in y]
    idx = np.round(np.linspace(0, len(y) - 1, n)).astype(int)
    return [round(float(y[i]), 4) for i in idx]


# ── Core pitch pipeline ──────────────────────────────────────────────────────

def _smooth_f0(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """
    Median-filter f0 in MIDI space within each voiced segment independently.
    This removes single-frame octave flips and pitch jitter while preserving
    genuine pitch transitions between notes.
    """
    f0_work = f0.copy()
    labeled_arr, n_segs = label(voiced.astype(int))

    for seg_id in range(1, n_segs + 1):
        idx = np.where(labeled_arr == seg_id)[0]
        if len(idx) < 3:
            continue
        seg_hz = f0_work[idx]
        # Convert to MIDI (perceptually uniform scale)
        safe_hz = np.where(seg_hz > 0, seg_hz, 1e-6)
        seg_midi = librosa.hz_to_midi(safe_hz)
        size = min(7, len(idx))
        if size % 2 == 0:
            size -= 1
        seg_midi_filt = median_filter(seg_midi, size=size)
        f0_work[idx] = librosa.midi_to_hz(seg_midi_filt)

    return f0_work


def _correct_octave_errors(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """
    Find isolated short regions (≤ 5 frames) where the MIDI value differs by
    exactly 12 from both the preceding and following voiced regions. Those are
    almost certainly sub-harmonic or double-harmonic captures; shift them by
    ±1 octave back to match the surrounding context.
    """
    f0_work = f0.copy()
    midi_arr = np.where(voiced & (f0_work > 0),
                        np.round(librosa.hz_to_midi(np.where(f0_work > 0, f0_work, 440))).astype(int),
                        -1)

    labeled_arr, n_segs = label(voiced.astype(int))
    for seg_id in range(1, n_segs + 1):
        idx = np.where(labeled_arr == seg_id)[0]
        if len(idx) > 5:
            continue
        # Find neighboring voiced frames
        before = idx[0] - 1
        after  = idx[-1] + 1
        midi_seg = midi_arr[idx]
        if (before >= 0 and midi_arr[before] > 0 and
                after < len(midi_arr) and midi_arr[after] > 0):
            ctx = midi_arr[before]
            diff = int(round(np.median(midi_seg))) - ctx
            if abs(diff) == 12:
                f0_work[idx] = f0_work[idx] * (0.5 if diff > 0 else 2.0)

    return f0_work


def _build_note_events(f0: np.ndarray, voiced: np.ndarray,
                       voiced_probs: np.ndarray,
                       sr: int, bpm: float) -> list[NoteEvent]:
    """
    Merge consecutive voiced frames with similar pitch into NoteEvent objects.
    """
    times     = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)
    frame_dur = float(times[1] - times[0]) if len(times) > 1 else HOP_LENGTH / sr

    notes = []
    i = 0
    while i < len(f0):
        if not voiced[i]:
            i += 1
            continue

        # Fixed anchor at the first frame's rounded MIDI value.
        # Using a fixed (not sliding) anchor prevents gradual drift from
        # accidentally bridging two adjacent semitones into one note.
        midi_anchor = float(round(librosa.hz_to_midi(max(f0[i], 1e-6))))
        j = i + 1
        while j < len(f0) and voiced[j]:
            midi_j = float(librosa.hz_to_midi(max(f0[j], 1e-6)))
            if abs(midi_j - midi_anchor) * 100 > MERGE_CENTS:
                break
            j += 1

        duration = (j - i) * frame_dur
        if duration < MIN_NOTE_DUR:
            i = j
            continue

        segment_hz    = f0[i:j]
        segment_probs = voiced_probs[i:j]
        mean_hz       = float(np.mean(segment_hz))
        mean_conf     = float(np.mean(segment_probs))
        midi_int      = int(round(librosa.hz_to_midi(mean_hz)))
        midi_int      = max(24, min(midi_int, 108))  # C1 – C8 guard

        note_str, pc, oct = _midi_to_note(midi_int)
        cents = _cents_offset(mean_hz, midi_int)
        t_start = float(times[i])
        t_end   = float(times[min(j, len(times)-1)])
        beat_start = (t_start * bpm) / 60.0
        beat_dur   = (duration * bpm) / 60.0

        notes.append(NoteEvent(
            note=note_str,
            pitch_class=pc,
            octave=oct,
            midi=midi_int,
            frequency_hz=round(mean_hz, 2),
            cents_offset=cents,
            start_sec=round(t_start, 4),
            end_sec=round(t_end, 4),
            duration_sec=round(duration, 4),
            start_beat=round(beat_start, 3),
            duration_beats=round(beat_dur, 3),
            confidence=round(mean_conf, 3),
        ))
        i = j

    return notes


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_audio(audio_bytes: bytes, filename: str = 'audio.webm') -> dict:
    """
    Detect monophonic notes from raw audio bytes.
    Returns a JSON-serialisable dict.
    """
    suffix = os.path.splitext(filename)[-1].lower() or '.webm'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=SR, mono=True, duration=90)
    except Exception as exc:
        return {'error': f'Could not decode audio: {exc}'}
    finally:
        os.unlink(tmp_path)

    if len(y) < sr * 0.1:
        return {'error': 'Recording too short (< 0.1 s)'}

    duration = len(y) / sr

    # ── Tempo ────────────────────────────────────────────────────────────────
    try:
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo_arr)[0])
        bpm = round(bpm if bpm > 0 else 120.0, 1)
    except Exception:
        bpm = 120.0

    # ── Harmonic isolation ───────────────────────────────────────────────────
    # margin=8 gives aggressive harmonic/percussive separation; helps on
    # plucked or struck instruments where transients muddy the pitch estimate.
    y_harm, _ = librosa.effects.hpss(y, margin=8)

    # ── pyin pitch tracking ──────────────────────────────────────────────────
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_harm,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    except Exception as exc:
        return {'error': f'Pitch detection failed: {exc}'}

    # Replace NaN with 0 for arithmetic safety
    f0 = np.where(voiced_flag & np.isfinite(f0), f0, 0.0)

    # ── Strict confidence gate ───────────────────────────────────────────────
    voiced_strict = voiced_flag & (voiced_probs >= VOICED_THRESH)

    # ── Smooth and correct ───────────────────────────────────────────────────
    f0_smooth    = _smooth_f0(f0, voiced_strict)
    f0_corrected = _correct_octave_errors(f0_smooth, voiced_strict)

    # ── Build note events ────────────────────────────────────────────────────
    notes = _build_note_events(f0_corrected, voiced_strict, voiced_probs, sr, bpm)

    # ── Waveform for display ─────────────────────────────────────────────────
    waveform = _downsample(y)

    return {
        'notes':           [vars(n) for n in notes],
        'note_count':      len(notes),
        'duration_sec':    round(duration, 3),
        'tempo_bpm':       bpm,
        'sample_rate':     sr,
        'waveform_samples': waveform,
    }
