# Web UI

Next.js 16 App Router frontend. Server routes under `app/api/*` proxy read-only requests to the Flask API via `API_BASE_URL`.

---

## Local development

See **[docs/LOCAL_TESTING.md](../docs/LOCAL_TESTING.md)** for the full checklist (CLI refresh, smoke curls, UI verification, troubleshooting).

```bash
# Terminal 1 — from repository root
source .venv/bin/activate
python app/server.py --port 8000

# Terminal 2 — from web/
cd web
cp .env.example .env.local   # API_BASE_URL=http://127.0.0.1:8000
npm install && npm run dev
```

Open http://localhost:3000

```bash
curl -s http://127.0.0.1:8000/api/data-status | python3 -m json.tool
```

---

## Routes

| Route | Description |
|-------|-------------|
| `/` | Markets — ticker grid, ✓/✗ resolved calls, overview chart |
| `/t/[symbol]` | Ticker detail — close-to-close hero, resolved strip (7/30/90d), Why / Advanced, charts |
| `/status` | Data freshness and evaluation summary |

Global date picker syncs across pages.

---

## API proxies

Handlers in `app/api/*/route.ts` forward to Flask. Most upstream routes live under `/api/*`; health checks use `/healthz` at the Flask root.

Mutation routes (`/api/run`, `/api/train`) are not exposed — training is CLI-only.

Key client modules: `lib/backend.ts`, `lib/types.ts`, `lib/models.ts`.

---

## Branding

| File | Purpose |
|------|---------|
| `app/icon.png` | Favicon (512×512) |
| `app/apple-icon.png` | Apple touch icon |
| `app/favicon.ico` | Legacy browsers |
| `app/layout.tsx` | Page title and metadata |

---

## Deploy (Vercel)

1. **Root Directory:** `web`
2. **Environment:** `API_BASE_URL=https://<railway-api-domain>` (no trailing slash)
3. Custom domain optional

```bash
npm run build
npm run lint
```

Deployment details: [docs/DEPLOY.md](../docs/DEPLOY.md) (operator doc).
