# Musician App – Audio to Music Notes

Record or upload audio and get back a full musical transcription: notes, chords, key signature, overtone series, and frequency-band source separation — all in your browser.

---

## Features

| Feature | Details |
|---|---|
| **Note detection** | Probabilistic YIN (pyin) pitch tracking on the harmonic signal component |
| **Key signature** | Krumhansl–Schmuckler algorithm across all 24 major/minor profiles |
| **Chord identification** | 110 chord templates (maj, min, 7, maj7, min7, dim, aug, sus2, sus4, dim7, add9) matched via chroma features |
| **Overtone series** | FFT peak detection with harmonic partial labeling (×1 – ×8) and relative amplitude bars |
| **Source separation** | Butterworth bandpass filters split audio into three functional bands: **Bass** (20–320 Hz), **Mid / Chords** (180–2200 Hz), and **Lead / Melody** (1400 Hz+) |
| **Tempo detection** | Beat tracking via librosa |

---

## Requirements

- Python 3.10+
- ffmpeg (for broad audio format support)

Install ffmpeg on Ubuntu/Debian:

```bash
sudo apt-get install ffmpeg
```

On macOS (with Homebrew):

```bash
brew install ffmpeg
```

---

## Running the App

```bash
./run.sh
```

This installs Python dependencies (if needed) and starts the server. Then open **http://localhost:8000** in your browser.

To use a different port:

```bash
./run.sh 9000
```

To install dependencies and start manually:

```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## How to Use

### 1. Record audio

Click the **red circle button** to start recording from your microphone. A live waveform will appear as you play. Click the button again to stop.

For best results:
- Play in a quiet environment
- Hold notes long enough for the pitch detector to lock on (≥ 0.5 s per note)
- Recordings up to 90 seconds are supported

### 2. Upload a file

Click **Choose file** to upload an existing audio file instead of recording. WAV, MP3, FLAC, OGG, and WebM are all supported.

### 3. Analyze

Click **Analyze**. Processing typically takes 5–20 seconds depending on recording length.

### 4. Read the results

**Summary badges** — key signature, tempo (BPM), duration, and key confidence percentage appear at the top.

**Piano keyboard** — highlights the detected key root (blue-purple), notes that belong to the scale (light blue), and notes that were actually played (solid blue/purple).

**Track tabs** — switch between four views, each showing its own waveform, note piano roll, note list, chord progression, and scale notes:

| Tab | Frequency range | Best for |
|---|---|---|
| Full Mix | 20 Hz – 20 kHz | Overall picture |
| Bass | 20 – 320 Hz | Bass lines, low fundamentals |
| Mid | 180 – 2200 Hz | Chords, rhythm guitar, piano, voices |
| Lead | 1400 Hz+ | Melody, lead guitar, high vocals |

**Overtone panel** — shows the strongest harmonic series found in the recording. Each fundamental lists its partials (×1 fundamental, ×2 octave, ×3 perfect fifth + octave, etc.) with relative amplitude bars. This reveals the timbre and helps identify which instruments are present.

---

## Project Structure

```
musician-app-poc/
├── analyzer.py       # DSP engine: pitch, chord, key, overtone, separation
├── main.py           # FastAPI server (API + static file serving)
├── static/
│   └── index.html    # Single-page frontend (vanilla JS / CSS)
├── requirements.txt  # Python dependencies
└── run.sh            # Convenience start script
```

---

## API

The backend exposes two endpoints:

```
POST /api/analyze
```
Upload an audio file (`multipart/form-data`, field name `file`). Returns JSON with:

```jsonc
{
  "key": "A major",
  "key_confidence": 0.868,
  "tempo_bpm": 120.0,
  "duration_sec": 3.0,
  "waveform_samples": [...],   // 800-point downsampled waveform
  "tracks": {
    "full":   { "notes": [...], "chords": [...], "overtones": [...], ... },
    "bass":   { "notes": [...], "chords": [...], ... },
    "mid":    { "notes": [...], "chords": [...], ... },
    "treble": { "notes": [...], "chords": [...], ... }
  }
}
```

```
GET /api/health
```
Returns `{"status": "ok"}`.

---

## Limitations

- **Polyphonic note detection** is approximate. The pyin algorithm is designed for monophonic signals; it tracks the dominant pitch within each frequency band. For dense chords, chord detection (chroma-based) is more reliable than the individual note list.
- **Source separation** uses frequency band filtering, not deep learning. Bleed between bands is normal for instruments with wide frequency ranges (e.g. piano, voice).
- **Tempo** defaults to 120 BPM for recordings with no clear beat.
- Maximum recording length is 90 seconds; maximum upload size is 50 MB.
