# Agent Guidance

## Hosting Constraints — Free Tier Only

This project must remain deployable on **free tiers** of cloud providers.
Do not introduce services, libraries, or architectures that require a paid plan.

### Backend (Railway / Render free tier)
- No persistent volumes — use the local filesystem (`/tmp` or a process-level temp dir) for ephemeral storage only.
- No external databases (Postgres, Redis, etc.) unless they have a genuinely free tier that fits within the app's usage.
- Background tasks must be in-process (FastAPI `BackgroundTasks` or `asyncio`) — do not add Celery, RQ, or any worker queue that requires a separate paid process.
- Keep memory usage low: avoid loading large ML models unless they fit comfortably in the free tier RAM limit (~512 MB on Render free, ~512 MB on Railway Starter).
- Free tier services sleep after inactivity — the app must handle cold starts gracefully.

### Frontend (GitHub Pages)
- Must be a fully static build (no SSR, no server-side rendering).
- Build output must go into `frontend/dist/` so the GitHub Actions workflow can pick it up.
- No paid CDN or asset hosting — all assets served from GitHub Pages.

### Storage
- No S3, GCS, or any object storage (all have costs at scale). Use ephemeral local disk only.
- Videos and results are temporary; a background task must clean them up (default: 1 hour TTL).

### General
- Prefer libraries available on PyPI / npm with permissive licenses.
- Do not add paid API calls (OpenAI, cloud vision APIs, etc.) as a core dependency — keep processing local.
