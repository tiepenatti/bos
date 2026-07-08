# BOS — Browser OpenCV Studio

A web application that lets users upload a video, run server-side OpenCV processing,
and download results (annotated frames, extracted clips, JSON analytics).

## Goals

- **Upload a video** from the browser (short clips suited for analysis)
- **Process with OpenCV** on the backend (frame extraction, edge detection, object tracking, etc.)
- **Browse results** in the browser — individual frames or a JSON summary
- **Download** results: specific frames as images or the full result set as a ZIP
- **Ephemeral storage** — uploaded videos and results are held temporarily for download, then cleaned up automatically

## Stack

| Layer | Technology |
|---|---|
| Frontend | Vite + Vanilla JS, hosted on GitHub Pages |
| Backend | FastAPI (Python) + OpenCV |
| Hosting (BE) | Railway / Render free tier |
| Storage | In-process temp directory (ephemeral) |

## Project Structure

```
bos/
├── frontend/       # Static site (Vite)
│   └── src/
├── backend/        # FastAPI server
│   ├── main.py
│   └── requirements.txt
└── .github/
    └── workflows/  # CI/CD — deploy frontend to GitHub Pages
```

## Local Development

### Backend

```bash
cd backend
uv sync
uv run uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`. API docs available at `/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

Set `VITE_API_URL=http://localhost:8000` in `frontend/.env.local` to point the UI at your local backend.

## Deployment

- **Frontend**: pushed to `main` → GitHub Actions builds with Vite → deploys to GitHub Pages automatically.
- **Backend**: deploy the `backend/` folder to Railway or Render. Set the `FRONTEND_ORIGIN` environment variable to your GitHub Pages URL to enable CORS.

## Video Retention Policy

Uploaded videos and their results are stored in a temporary directory and deleted
after **1 hour** by a background cleanup task. No video data is persisted to any
external database or storage service.
