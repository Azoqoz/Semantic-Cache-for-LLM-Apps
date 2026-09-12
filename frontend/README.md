# CACHE FLOW

Next.js App Router + TypeScript. A light, responsive interface for the existing
FastAPI service. Cache logic, embedding inference and provider credentials stay
on the backend. Theme choice and an opaque demo cookie are retained in the browser.

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
It forwards only the opaque `cache_flow_demo` sandbox cookie, never authorization
headers or unrelated cookies, and does not log request bodies or
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
already be prefetched (`python -m src.prefetch_model`). From the repository root:

```powershell
.venv/Scripts/python.exe frontend/scripts/qa_backend.py
```

With `npm start` running on port 3000, run:

```powershell
node frontend/scripts/smoke.mjs
```

Run the smoke script against that temporary QA backend. Public Demo checks the
curated scenarios, cookie forwarding, new-visitor isolation, restrictions and
fixed evaluation. For the Local Mode checks, restart the QA backend with
`--mode local`; those checks clear only the disposable QA database. Stop it afterward.
Use `--port` and `SMOKE_BASE_URL` for isolated test ports if your main servers are running.

Public Demo presents the backend-curated request library in the existing Query Lab.
Free-form typing, settings changes and clearing are unavailable; every result still
comes from the real backend. Each visitor has an isolated, seeded SQLite sandbox
identified by an HttpOnly cookie. New intent misses once and becomes an exact hit
on repetition; other visitors do not change that starting state. Idle sandboxes
expire after an hour. Demo is fixed to provider Demo, model demo-rule-based,
threshold 0.84, TTL 0 and provider/model isolation. Cache Quality evaluates only
the existing labeled dataset at the fixed threshold.

Local Mode retains free-form prompts, every provider/model control, threshold,
TTL, isolation, inspection, clearing and configurable evaluation. Mode and controls
come from `/capabilities`, not hostname detection. Themes and component structure
are shared. The demo cookie is not authentication; no arbitrary/private user data
or external credentials are accepted in the public mode.

The frontend polls /ready every two seconds after liveness succeeds, stopping on ready or error. While warming, inference actions are disabled; model failures are distinct from network errors. Check readiness only reads status; an operator must restart the backend to retry a failed warm-up.
