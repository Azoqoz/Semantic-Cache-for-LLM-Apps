# CACHE FLOW

Next.js App Router + TypeScript. A light, responsive interface for the existing
FastAPI service. No cache logic, embedding inference, credentials, or persistent
browser storage are implemented in the frontend.

## Run

Start the existing backend from the repository root:

```powershell
.venv/Scripts/python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Then in `frontend/`:

```powershell
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. Use `npm run lint` and `npm run build` for validation,
then `npm start` for the production server. The lockfile records installed versions.

## Deploy

Set Vercel's root directory to `frontend`, use its Next.js preset, and configure
`NEXT_PUBLIC_API_BASE_URL` to the reachable HTTPS FastAPI origin (optionally with
a base path). The localhost example only works when both services run locally.
Provider credentials stay exclusively in the backend environment. Do not put
them in a `NEXT_PUBLIC_` variable. Hosted visitor use requires the backend's
`APP_MODE=demo`; mode is read from capabilities, never overridden in the browser.

The allowlisted `/api/backend/*` route handler proxies only the seven supported
operations. Browser calls are same-origin, so no backend CORS changes are needed.
It forwards no cookies or authorization headers, does not log request bodies or
upstream errors, and never caches API results. Backend connectivity errors become
safe messages. The 180-second request timeout also depends on hosting-plan limits;
timed-out work may finish on the backend. Refresh before repeating a mutation.

## What is displayed

- Request journey: final confirmed route, including embeddings for exact hits.
  A pending request never pretends to expose intermediate stages.
- Exact / semantic / miss: uses `hit_type` only. A successful miss confirms the
  existing pipeline's cache write; failures never display a completed path.
- Calibrated cosine ruler: signed scores when present, otherwise an explicit
  unavailable state. An exact match never fabricates a cosine score of 1.
- Query settings: provider/model, inclusive threshold, whole-hour TTL (zero or
  negative means no expiry), provider/model isolation. All decisions are backend-owned.
- Value: persisted hit counts, hit rate, supplied latency reduction and estimated
  avoided cost. No invented exact/semantic histogram or per-request time savings.
- Ledger: latest 1,000 entries, local text filtering, expandable answers, age,
  backend expiry timestamp, reuse count, refresh, and confirmed clear-all. Expiry
  labels use browser time for presentation only; eligibility remains backend-owned.
- Quality: independent evaluation threshold, accuracy/precision/recall/F1,
  Easy/Medium/Hard accuracy, confusion counts, threshold comparison and all pairs.
  Changing the slider leaves the previous report explicitly labeled until rerun.

API liveness is not provider readiness. Errors distinguish disconnection from
provider/configuration failure. Responses and cache entries render as plain text.
Native dialogs provide keyboard focus containment; inputs are labeled; status
announcements, visible focus, reduced motion and small-screen layouts are included.

## Isolated integration verification

The QA backend uses real embeddings, the real Demo provider and a temporary
SQLite database, leaving the user's database untouched. The embedding model must
already be available or downloadable. From the repository root:

```powershell
.venv/Scripts/python.exe frontend/scripts/qa_backend.py
```

With `npm start` running on port 3000, run:

```powershell
node frontend/scripts/smoke.mjs
```

Run the smoke script only against that temporary QA backend: it clears all cache
data during verification. It covers health, capabilities, Demo miss/exact/semantic,
provider restrictions, ledger, evaluation, and clear. Stop the QA backend afterward.

The existing backend has no authentication or tenant isolation. Public deployment
shares cache records and the clear operation across visitors; deploy only data and
providers intended for public access. Existing SQLite concurrency, cost-estimation,
and synchronous inference limitations remain unchanged.

The frontend polls /ready every two seconds after liveness succeeds, stopping on ready or error. While warming, inference actions are disabled; model failures are distinct from network errors. Check readiness only reads status; an operator must restart the backend to retry a failed warm-up.
