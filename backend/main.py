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
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from bos import BOSParams, compute_schlieren

# ── Config ────────────────────────────────────────────────────────────────────

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", 900))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 200 * 1024 * 1024))

# Number of frame *pairs* to process evenly across the video
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


# ── Request models ─────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    """BOS pipeline parameters plus frame-sampling controls. Every field is
    optional so `POST /api/process/{job_id}` works with an empty body."""

    # Motion estimation
    method: str = "features"                # "features" | "optical_flow"
    detector: str = "orb"                   # "orb" | "akaze" | "sift"
    max_features: int = 2000
    match_ratio: float = 0.75
    min_matches: int = 12
    flow_max_corners: int = 600
    flow_quality: float = 0.01
    flow_min_distance: float = 7.0
    flow_win_size: int = 21
    ransac_thresh: float = 4.0

    # Schlieren post-processing
    blur_ksize: int = 5
    gain: float = 4.0
    threshold: int = 8
    morph_ksize: int = 3
    normalize: bool = True
    colormap: str = "inferno"
    overlay: float = 0.0
    border_erode: int = 5

    # Frame sampling
    num_pairs: int = Field(default=FRAMES_TO_EXTRACT, ge=1, le=200)
    frame_step: int = Field(default=1, ge=1, le=100)  # gap between prev & curr

    def to_bos_params(self) -> BOSParams:
        keys = {
            "method", "detector", "max_features", "match_ratio", "min_matches",
            "flow_max_corners", "flow_quality", "flow_min_distance", "flow_win_size",
            "ransac_thresh", "blur_ksize", "gain", "threshold", "morph_ksize",
            "normalize", "colormap", "overlay", "border_erode",
        }
        return BOSParams(**{k: getattr(self, k) for k in keys})


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
def process_video(
    job_id: str,
    params: ProcessRequest = Body(default_factory=ProcessRequest),
) -> dict:
    """Run the Background-Oriented Schlieren pipeline on consecutive frame pairs.

    For each sampled anchor frame we grab the frame `frame_step` positions
    earlier, align the two (cancelling camera motion), take their absdiff, and
    post-process the result into a schlieren visualization."""
    d = require_job(job_id)

    # Find the input video
    input_files = list(d.glob("input.*"))
    if not input_files:
        raise HTTPException(status_code=404, detail="Input video not found")
    video_path = str(input_files[0])

    bos_params = params.to_bos_params()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Could not open video file")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = (frame_count / fps) if fps > 0 else 0

        # Clear any prior run's frames so stale results don't linger.
        frames_dir = d / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(exist_ok=True)

        step = params.frame_step
        # Select evenly-spaced anchor (current) frame indices. Each anchor must
        # have a valid previous frame `step` positions earlier.
        n = min(params.num_pairs, max(0, frame_count - step))
        if n > 0:
            span = frame_count - step
            anchors = [step + int(i * span / n) for i in range(n)]
        else:
            anchors = []

        extracted: list[dict[str, Any]] = []
        for idx in anchors:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx - step)
            ok_prev, prev_frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok_curr, curr_frame = cap.read()
            if not ok_prev or not ok_curr:
                continue

            vis, stats = compute_schlieren(prev_frame, curr_frame, bos_params)

            out_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), vis)
            timestamp_ms = (idx / fps * 1000) if fps > 0 else 0
            extracted.append(
                {
                    "index": idx,
                    "prev_index": idx - step,
                    "timestamp_ms": round(timestamp_ms, 1),
                    "stats": stats,
                }
            )

    finally:
        cap.release()

    results = {
        "job_id": job_id,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(duration_seconds, 2),
        "params": params.model_dump(),
        "extracted_frames": extracted,
        "processed_at": time.time(),
    }
    (d / "results.json").write_text(json.dumps(results))
    return results


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
