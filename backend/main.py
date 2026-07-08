"""
BOS Backend — FastAPI + OpenCV
Handles video upload, processing, frame extraction, and result download.
All storage is ephemeral (local tmp dir); jobs are cleaned up after JOB_TTL_SECONDS.
"""

import asyncio
import contextlib
import io
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ── Config ────────────────────────────────────────────────────────────────────

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", 3600))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 200 * 1024 * 1024))

# Number of frames to extract evenly across the video
FRAMES_TO_EXTRACT = int(os.getenv("FRAMES_TO_EXTRACT", 12))

TMP_ROOT = Path(tempfile.gettempdir()) / "bos_jobs"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
print(f"[bos] temp storage: {TMP_ROOT}")

# ── Job helpers ───────────────────────────────────────────────────────────────

def job_dir(job_id: str) -> Path:
    return TMP_ROOT / job_id


def require_job(job_id: str) -> Path:
    d = job_dir(job_id)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return d


async def delete_job_after_ttl(job_id: str, ttl: int = JOB_TTL_SECONDS) -> None:
    """Scheduled via BackgroundTasks to clean up job files after TTL.
    Uses asyncio.sleep so uvicorn can cancel it cleanly on reload/shutdown.
    If the server is killed before this runs, cleanup happens on next startup instead."""
    await asyncio.sleep(ttl)
    d = job_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def cleanup_stale_jobs() -> None:
    """Delete any job directories older than JOB_TTL_SECONDS. Called on startup
    so files left behind by a previous server run are always cleaned up."""
    cutoff = time.time() - JOB_TTL_SECONDS
    for d in TMP_ROOT.iterdir():
        if not d.is_dir():
            continue
        meta_file = d / "meta.json"
        try:
            uploaded_at = json.loads(meta_file.read_text()).get("uploaded_at", 0)
        except Exception:
            uploaded_at = d.stat().st_mtime
        if uploaded_at < cutoff:
            shutil.rmtree(d, ignore_errors=True)


# ── App ───────────────────────────────────────────────────────────────────────

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_stale_jobs()
    yield


app = FastAPI(title="BOS API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so unhandled errors return JSON with CORS headers
    instead of propagating past the CORS middleware as bare 500s."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_video(
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> dict:
    """Accept a video file upload and store it temporarily. Returns a job_id."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB",
        )

    job_id = str(uuid.uuid4())
    d = job_dir(job_id)
    d.mkdir(parents=True)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    video_path = d / f"input{suffix}"
    video_path.write_bytes(content)

    # Store metadata
    (d / "meta.json").write_text(
        json.dumps({"filename": file.filename, "size": len(content), "uploaded_at": time.time()})
    )

    # Schedule cleanup
    background_tasks.add_task(delete_job_after_ttl, job_id)

    return {"job_id": job_id}


@app.post("/api/process/{job_id}")
def process_video(job_id: str) -> dict:
    """Run OpenCV processing on the uploaded video and save results + frames."""
    d = require_job(job_id)

    # Find the input video
    input_files = list(d.glob("input.*"))
    if not input_files:
        raise HTTPException(status_code=404, detail="Input video not found")
    video_path = str(input_files[0])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open video file")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = (frame_count / fps) if fps > 0 else 0

        frames_dir = d / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Select evenly-spaced frame indices to extract
        n = min(FRAMES_TO_EXTRACT, frame_count)
        indices = [int(i * frame_count / n) for i in range(n)] if n > 0 else []

        extracted: list[dict[str, Any]] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # ── OpenCV processing per frame (extend this section) ──────────────
            processed = apply_processing(frame)
            # ───────────────────────────────────────────────────────────────────

            out_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), processed)
            timestamp_ms = (idx / fps * 1000) if fps > 0 else 0
            extracted.append({"index": idx, "timestamp_ms": round(timestamp_ms, 1)})

    finally:
        cap.release()

    results = {
        "job_id": job_id,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(duration_seconds, 2),
        "extracted_frames": extracted,
        "processed_at": time.time(),
    }
    (d / "results.json").write_text(json.dumps(results))
    return results


def apply_processing(frame: np.ndarray) -> np.ndarray:
    """
    Apply OpenCV processing to a single frame.
    Extend or replace this function with your desired pipeline.

    Current default: Canny edge detection overlaid on the original frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    # Build a green-channel-only BGR image from the edge mask (stays uint8)
    empty = np.zeros_like(edges)
    edges_green = cv2.merge([empty, edges, empty])
    # Blend edges (green tint) with original
    overlay = cv2.addWeighted(frame, 0.7, edges_green, 0.3, 0)
    return overlay


@app.get("/api/results/{job_id}")
def get_results(job_id: str) -> dict:
    """Return the JSON processing results for a job."""
    d = require_job(job_id)
    results_file = d / "results.json"
    if not results_file.exists():
        raise HTTPException(status_code=404, detail="Results not ready. Call POST /api/process first.")
    return json.loads(results_file.read_text())


@app.get("/api/frames/{job_id}/{frame_index}")
def get_frame(job_id: str, frame_index: int) -> FileResponse:
    """Return a specific extracted frame as a JPEG image."""
    d = require_job(job_id)
    frame_path = d / "frames" / f"frame_{frame_index:06d}.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@app.get("/api/download/{job_id}")
def download_zip(job_id: str) -> StreamingResponse:
    """Stream a ZIP archive containing all frames and the results JSON."""
    d = require_job(job_id)

    def zip_generator():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            results_file = d / "results.json"
            if results_file.exists():
                zf.write(results_file, "results.json")
            frames_dir = d / "frames"
            if frames_dir.exists():
                for frame_file in sorted(frames_dir.glob("*.jpg")):
                    zf.write(frame_file, f"frames/{frame_file.name}")
        buf.seek(0)
        yield buf.read()

    return StreamingResponse(
        zip_generator(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=bos_results_{job_id[:8]}.zip"},
    )
