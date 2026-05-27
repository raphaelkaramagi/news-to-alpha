# News-to-Alpha Web UI

Next.js 16 App Router frontend for the News-to-Alpha API.

## Local development

```bash
# Terminal 1 — from repo root
source .venv/bin/activate
python app/server.py --port 8000

# Terminal 2 — must be in web/ (no package.json at repo root)
cd web
cp .env.example .env.local   # API_BASE_URL=http://127.0.0.1:8000
npm install
npm run dev
```

Open http://localhost:3000

If port 8000 is busy: `lsof -nP -iTCP:8000 -sTCP:LISTEN` → kill the old Flask process.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Markets — 15-ticker grid, outcome dots/legend, overview chart |
| `/t/[symbol]` | Ticker detail — call, headlines (top 3 + expand), Why tab, Advanced, synced charts |
| `/status` | Data freshness |

Global date picker in the header syncs Markets + ticker views.

## API proxies

Read-only routes under `app/api/*` proxy to Flask (`API_BASE_URL`):

- `ticker`, `data-status`, `dates`, `headlines`, `rationale`, `history`
- `last-resolved`, `accuracy-summary`, `accuracy-trace`, `metrics`, `conviction`
- `markets-overview`, `healthz`

`markets-overview` falls back to client-side aggregation if Flask lacks the route.

Mutation routes (`/api/run`, `/api/train`) are **not** exposed — training is CLI-only.

## Stack

- Tailwind CSS + zinc theme
- TanStack Query (client cache)
- Recharts (price + P(UP) chart)

## Deploy (Vercel)

1. Root directory: **`web`**
2. Env: `API_BASE_URL=https://your-railway-app.up.railway.app` (no trailing slash)
3. Deploy

See [../docs/DEPLOY_UI.md](../docs/DEPLOY_UI.md) for the full guide.

## Build

```bash
npm run build
npm run lint
```
