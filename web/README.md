# Web UI

Next.js 16 App Router frontend. Server routes under `app/api/*` proxy read-only requests to the Flask API via `API_BASE_URL`.

---

## Local development

See **[docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md)** for environment setup and smoke tests.

```bash
# Terminal 1 — repository root
source .venv/bin/activate
python app/server.py --port 8000

# Terminal 2 — web/
cd web
cp .env.example .env.local   # API_BASE_URL=http://127.0.0.1:8000
npm install && npm run dev
```

Open http://localhost:3000

---

## Routes

| Route | Description |
|-------|-------------|
| `/` | Markets — direction call, expected move (±%), overview |
| `/t/[symbol]` | Hero band, resolved strip, Why / Advanced (vol calibration) |
| `/status` | Data freshness and evaluation summary (`news_scored` metrics when available) |

Global date picker syncs across pages.

---

## API proxies

Handlers in `app/api/*/route.ts` forward to Flask. Most upstream routes live under `/api/*`; health checks use `/healthz` at the Flask root.

Mutation routes (`/api/run`, `/api/train`) are not exposed — training is CLI-only.

Key client modules: `lib/backend.ts`, `lib/types.ts`, `lib/models.ts`.

---

## Deploy (Vercel)

1. **Root Directory:** `web`
2. **Environment:** `API_BASE_URL=https://<api-host>` (no trailing slash)
3. Custom domain optional

```bash
npm run build
npm run lint
```

Backend deployment and artifact publishing: [docs/DATA.md](../docs/DATA.md).
