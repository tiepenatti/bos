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
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
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


# In-memory progress for in-flight processing jobs. Single-process free tier, so
# a plain dict (guarded by the GIL for these simple assignments) is enough. Lost
# on restart — the status endpoint falls back to results.json in that case.
PROGRESS: dict[str, dict[str, Any]] = {}


def set_progress(job_id: str, **fields: Any) -> None:
    entry = PROGRESS.setdefault(job_id, {})
    entry.update(fields)


async def delete_job_after_ttl(job_id: str, ttl: int = JOB_TTL_SECONDS) -> None:
    """Scheduled via BackgroundTasks to clean up job files after TTL.
    Uses asyncio.sleep so uvicorn can cancel it cleanly on reload/shutdown.
    If the server is killed before this runs, cleanup happens on next startup instead."""
    await asyncio.sleep(ttl)
    d = job_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    PROGRESS.pop(job_id, None)


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

    # Frame sampling / output
    # The whole video is processed frame-by-frame into a playable clip.
    # `num_pairs` only controls how many still thumbnails are saved for the grid.
    num_pairs: int = Field(default=FRAMES_TO_EXTRACT, ge=1, le=200)
    frame_step: int = Field(default=1, ge=1, le=100)  # gap between prev & curr
    max_width: int = Field(default=960, ge=0, le=3840)  # downscale cap; 0 = original

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


def _scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Return (w, h) downscaled so width <= max_width (even dimensions for the
    video encoder). max_width <= 0 keeps the original size."""
    if max_width <= 0 or width <= max_width:
        w, h = width, height
    else:
        scale = max_width / width
        w, h = int(round(width * scale)), int(round(height * scale))
    # VP8 needs even dimensions.
    return (w - (w % 2)), (h - (h % 2))


@app.post("/api/process/{job_id}")
async def process_video(
    job_id: str,
    params: ProcessRequest = Body(default_factory=ProcessRequest),
) -> dict:
    """Kick off the Background-Oriented Schlieren pipeline as an in-process
    background task and return immediately. Poll GET /api/status/{job_id} for
    progress, then GET /api/results/{job_id} once it reports done.

    The video is streamed frame-by-frame: each frame is aligned against the frame
    `frame_step` positions earlier (cancelling camera motion), diffed, and
    post-processed into a schlieren frame written to an output .webm."""
    d = require_job(job_id)

    input_files = list(d.glob("input.*"))
    if not input_files:
        raise HTTPException(status_code=404, detail="Input video not found")

    # Don't start a second run while one is already in flight for this job.
    existing = PROGRESS.get(job_id)
    if existing and existing.get("state") in ("queued", "running"):
        return {"status": "already_running", "job_id": job_id}

    set_progress(
        job_id,
        state="queued",
        percent=0,
        processed=0,
        total=None,
        error=None,
        started_at=time.time(),
    )
    # asyncio.create_task keeps the work in-process (free-tier friendly); the
    # blocking OpenCV loop runs in a worker thread so the event loop stays free.
    asyncio.create_task(asyncio.to_thread(_process_job, job_id, params))
    return {"status": "started", "job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str) -> dict:
    """Return processing progress for a job. Falls back to results.json when the
    in-memory entry is missing (e.g. after a server restart)."""
    d = require_job(job_id)
    entry = PROGRESS.get(job_id)
    if entry is not None:
        return {"job_id": job_id, **entry}
    if (d / "results.json").exists():
        return {"job_id": job_id, "state": "done", "percent": 100}
    return {"job_id": job_id, "state": "unknown", "percent": 0}


def _process_job(job_id: str, params: ProcessRequest) -> None:
    """Blocking worker: encode the full schlieren video, updating PROGRESS as it
    goes. Runs in a thread; never raises (errors are recorded in PROGRESS)."""
    d = job_dir(job_id)
    video_path = str(next(iter(d.glob("input.*"))))
    bos_params = params.to_bos_params()
    step = params.frame_step

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        set_progress(job_id, state="error", error="Could not open video file")
        return

    # Fresh output dir/files each run so stale results don't linger.
    frames_dir = d / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(exist_ok=True)
    video_out = d / "output.webm"
    if video_out.exists():
        video_out.unlink()

    writer: cv2.VideoWriter | None = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        out_fps = fps if fps and fps > 0 else 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = (frame_count / fps) if fps > 0 else 0

        out_w, out_h = _scaled_size(width, height, params.max_width)

        # Number of output frames we'll produce, and which of them to snapshot.
        total_out = max(0, frame_count - step)
        n_thumbs = min(params.num_pairs, total_out)
        thumb_positions = (
            {int(i * (total_out - 1) / max(1, n_thumbs - 1)) for i in range(n_thumbs)}
            if n_thumbs > 0
            else set()
        )
        set_progress(
            job_id,
            state="running",
            total=(total_out if total_out > 0 else None),
            processed=0,
            percent=0,
        )

        # Ring buffer holding the last (step + 1) frames; buf[0] is `step`
        # frames behind buf[-1], giving us the (prev, curr) pair.
        buf: deque[np.ndarray] = deque(maxlen=step + 1)
        extracted: list[dict[str, Any]] = []
        out_idx = 0  # index into the produced output frames
        src_idx = 0  # index into the source frames

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (out_w, out_h) != (width, height):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            buf.append(frame)

            if len(buf) == step + 1:
                prev_frame = buf[0]
                curr_frame = buf[-1]
                vis, stats = compute_schlieren(prev_frame, curr_frame, bos_params)

                if writer is None:
                    writer = cv2.VideoWriter(
                        str(video_out),
                        cv2.VideoWriter_fourcc(*"VP80"),
                        out_fps,
                        (out_w, out_h),
                    )
                    if not writer.isOpened():
                        set_progress(
                            job_id,
                            state="error",
                            error="Could not initialize video encoder (VP8/webm)",
                        )
                        return
                writer.write(vis)

                if out_idx in thumb_positions:
                    cur_src = src_idx
                    cv2.imwrite(str(frames_dir / f"frame_{out_idx:06d}.jpg"), vis)
                    timestamp_ms = (cur_src / fps * 1000) if fps > 0 else 0
                    extracted.append(
                        {
                            "index": out_idx,
                            "src_index": cur_src,
                            "prev_index": cur_src - step,
                            "timestamp_ms": round(timestamp_ms, 1),
                            "stats": stats,
                        }
                    )
                out_idx += 1
                # Update progress (cheap; every processed frame).
                if total_out > 0:
                    percent = min(99, int(out_idx * 100 / total_out))
                else:
                    percent = 0
                set_progress(job_id, processed=out_idx, percent=percent)
            src_idx += 1

    except Exception as exc:  # noqa: BLE001 — record any failure for the client
        set_progress(job_id, state="error", error=f"{exc}")
        return
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    has_video = video_out.exists() and video_out.stat().st_size > 0
    results = {
        "job_id": job_id,
        "fps": fps,
        "output_fps": out_fps,
        "frame_count": frame_count,
        "processed_frames": out_idx,
        "width": width,
        "height": height,
        "output_width": out_w,
        "output_height": out_h,
        "duration_seconds": round(duration_seconds, 2),
        "params": params.model_dump(),
        "has_video": has_video,
        "extracted_frames": extracted,
        "processed_at": time.time(),
    }
    (d / "results.json").write_text(json.dumps(results))
    set_progress(
        job_id, state="done", percent=100, processed=out_idx, total=out_idx
    )


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


@app.get("/api/video/{job_id}")
def get_video(job_id: str, request: Request) -> Response:
    """Serve the processed .webm, honoring HTTP Range requests so the browser
    <video> element can stream and seek."""
    d = require_job(job_id)
    video_path = d / "output.webm"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")
    media_type = "video/webm"

    if range_header is None:
        return FileResponse(
            str(video_path),
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    # Parse "bytes=start-end"
    try:
        units, _, rng = range_header.partition("=")
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Range header")

    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        raise HTTPException(status_code=416, detail="Requested range not satisfiable")

    chunk_size = end - start + 1
    with open(video_path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)

    return Response(
        content=data,
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
        },
    )


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
            video_file = d / "output.webm"
            if video_file.exists():
                zf.write(video_file, "schlieren.webm")
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
