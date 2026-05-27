# Stock Price and Sentiment Predictor — Web UI

Next.js 16 App Router frontend. Proxies read-only API calls to Flask via `API_BASE_URL`.

## Local development

```bash
# Terminal 1 — from repo root
source .venv/bin/activate
python app/server.py --port 8000

# Terminal 2 — must be in web/
cd web
cp .env.example .env.local   # API_BASE_URL=http://127.0.0.1:8000
npm install
npm run dev
```

Open http://localhost:3000

## Pages

| Route | Description |
|-------|-------------|
| `/` | Markets — 15-ticker grid, outcome dots, overview chart |
| `/t/[symbol]` | Ticker detail — call, headlines, Why tab, charts |
| `/status` | Data freshness |

## Icons & branding

Next.js App Router picks these up automatically (no manual `<link>` tags):

| File | Purpose |
|------|---------|
| `app/icon.png` | Favicon / tab icon (512×512) |
| `app/apple-icon.png` | Apple touch icon (180×180) |
| `app/favicon.ico` | Legacy browsers |
| `public/icon-192.png`, `icon-512.png` | PWA manifest (`app/manifest.ts`) |

Tab title: **Stock Price and Sentiment Predictor** — set in `app/layout.tsx`.

Replace `app/icon.png`, `app/apple-icon.png`, and `app/favicon.ico` directly to update branding.

## API proxies

Server routes under `app/api/*` forward to Flask. Most Flask routes live under `/api/*`; **`/healthz`** is at the root (see `app/api/healthz/route.ts`).

Mutation routes (`/api/run`, `/api/train`) are not exposed — training is CLI-only.

## Deploy (Vercel)

1. Root directory: **`web`**
2. Env: `API_BASE_URL=https://your-railway-api.up.railway.app` (no trailing slash)
3. Custom domain optional (e.g. `stock.yourdomain.com`)

## Build

```bash
npm run build
npm run lint
```
