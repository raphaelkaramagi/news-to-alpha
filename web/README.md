# News-to-Alpha — Next.js UI

This folder is the **Vercel-hosted** frontend. Heavy ML, SQLite, and pipelines stay in the parent Python project.

## Develop

```bash
cd web
cp .env.example .env.local
# set API_BASE_URL to your Flask origin (e.g. http://127.0.0.1:8000)
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Route Handlers under `app/api/*` proxy to `API_BASE_URL` so the browser never needs CORS or secrets.

## Deploy on Vercel

1. New project → import this repo.
2. **Root Directory:** `web`.
3. **Environment variables:** `API_BASE_URL` = public `https://…` origin of your Flask app (Railway/Render/Fly).

`npm run build` should succeed; pages that call the backend will show a warning if `API_BASE_URL` is missing.

## Stack

- Next.js App Router, TypeScript, Tailwind.
