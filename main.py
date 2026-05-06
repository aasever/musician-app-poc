"""
Musician App – FastAPI backend.
Serves the frontend and exposes /api/analyze for audio upload + analysis.
"""

import os
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analyzer import analyze_audio

app = FastAPI(title="Musician App", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

# ── API routes ───────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Accept an audio file and return full musical analysis as JSON."""
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty file")
    if len(audio_bytes) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50 MB)")

    result = analyze_audio(audio_bytes, filename=file.filename or "audio.webm")

    if "error" in result:
        raise HTTPException(422, result["error"])

    return JSONResponse(content=result)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Static files / SPA ───────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Musician App</h1><p>Frontend not found.</p>")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
