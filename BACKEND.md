# Migration backend

Install `requirements.txt`, then run locally:

```powershell
.venv/Scripts/python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Interactive contract: `/docs`; schema: `/openapi.json`. For tests install
`requirements-dev.txt` and run `python -B -m unittest discover -v`.

`CacheApplication` is independent of FastAPI. It delegates to the existing
`SemanticCacheService`, SQLite store, embedding service, provider factory, and
evaluation functions. FastAPI supplies input validation, serialization, and safe
errors. The existing Streamlit entry point is unchanged. Both entry points use
the same configured SQLite database by default.

| Endpoint | Contract |
| --- | --- |
| `GET /ready` | Reports `warming`, `ready`, or a safe `error` message. Polling does not retry initialization. |
| `GET /health` | Process liveness only; does not load embeddings or probe providers. |
| `GET /capabilities` | Mode, allowed providers and default models, request defaults, evaluation thresholds. Allowed does not mean installed/configured/reachable. |
| `POST /query` | JSON `question` required; optional `provider`, `model`, `threshold`, `ttl_hours`, `isolate_by_model`. Defaults come from settings; provider defaults to Demo and isolation to true. |
| `GET /cache?limit=100` | Latest entries (including expired rows), global legacy metrics, and nullable latency reduction. Limit is 1–1000. No embedding vectors. |
| `DELETE /cache` | Deletes all cache entries and query events; returns `{"cleared": true}`. |
| `POST /evaluation` | JSON `{}` or `{"threshold": 0.84}`; returns the existing evaluation report for the fixed 36-pair dataset. Does not change query defaults, cache, or events. |

Query responses contain `hit_type` (`exact`, `semantic`, `miss`), `cache_hit`,
question/answer, matched question, provider/model, threshold/TTL/isolation used,
latency in milliseconds, and cost metadata. Hit type is recorded at lookup, never
inferred from a rounded score. `similarity` is measured cosine only: null on exact
lookup or when no candidate was scored; signed on below-threshold misses. Existing
internal similarity values and event metrics remain unchanged (exact=1 and
negative miss scores clamped to zero). Even an exact request still generates an
embedding before lookup. Duplicate inserts remain append-only; access counts are
updated once per hit. Threshold is inclusive in [0,1]; TTL <=0 never expires, and
expiration equality excludes the row. Omitted/null TTL uses the configured default.

`estimated_cost_usd` is nullable and accompanied by `cost_kind` (incurred/avoided)
and `cost_basis` (provider_estimate/default_estimate/local_zero). Unknown provider
fallback costs are null in query responses. Aggregate cost metrics retain legacy
zero fallbacks and are explicitly labeled `legacy_estimates_not_billing`; old
events cannot establish actual billing or estimate completeness. Latency reduction
uses the existing dashboard formula only when both hit and miss observations and
a positive miss latency exist; otherwise it is null. Negative reductions are kept.

Demo mode allows only Demo. Local mode allows Demo, OpenAI, Claude, Gemini and
Ollama. Models can be overridden; Demo's model is fixed. Provider URLs and API
credentials are server configuration only. Request models reject extra fields,
including API keys; responses/validation errors never echo rejected values.
No request bodies, credentials, or upstream exception details are logged by this
backend. Provider errors do not trigger another provider or a Demo fallback.
Configure upstream server/SDK logging separately; do not enable credential/body
logging. Questions and answers are still persisted as application content.

Errors have `{"error": {"code": "...", "message": "..."}}`: invalid requests
422, disabled/unconfigured providers 400, operational failures 503, unexpected
adapter failures 500. Messages are fixed and exclude upstream exception details.
The handlers follow FastAPI's [exception handler mechanism](https://fastapi.tiangolo.com/tutorial/handling-errors/).

This is a local backend, without authentication, tenant isolation, rate limiting,
or CORS configuration. It retains SQLite's linear scan, non-atomic concurrent
miss behavior and separate entry/event writes. Resource initialization starts automatically in a background thread
once per application process; Render startup loads the build-prepared model locally without downloading it. Liveness is available without waiting. Query, evaluation, and cache operations return structured HTTP 503 errors until ready. A failed warm-up stays failed until an operator restarts the service; requests never trigger loading or retry downloads. Evaluation
is synchronous and recomputes embeddings. Tests use real SQLite and cosine math,
with model inference and providers mocked; they do not verify live provider uptime.

## Render model preparation

Use the repository root as the Render service root directory and this Build Command:

```sh
pip install -r requirements.txt && python -m src.prefetch_model
```

Keep the Start Command:

```sh
python -m uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

The prefetch module reads `settings.embedding_model` (currently
`sentence-transformers/all-MiniLM-L6-v2`), downloads the existing model using
SentenceTransformer, and saves all its modules, tokenizer, configuration, and
weights under `.model-cache/sentence-transformers--all-MiniLM-L6-v2` in the build
artifact. It verifies that artifact with the same offline loader used at runtime.
Build errors propagate and prevent deployment. An existing artifact is validated
and reused; an invalid artifact fails rather than silently downloading again.
Remove the invalid `.model-cache` artifact before rebuilding if needed.

The path is relative to the project root, not the shell working directory or a
build-only home cache. Do not mount a runtime disk over `.model-cache` or exclude
it from a custom deployment artifact. The directory is Git-ignored.

On [Render](https://render.com/docs/environment-variables), `RENDER=true` makes a
missing artifact an initialization error instead of allowing a runtime download.
A present artifact always loads with `local_files_only=True`, using the existing
[SentenceTransformer save/load contract](https://www.sbert.net/docs/package_reference/sentence_transformer/index.html).
Local development without an artifact retains its original model loading behavior.

FastAPI still binds without waiting; background startup loads the cached weights
into memory once, then `/ready` becomes `ready`. Prefetch removes download time,
not Python/PyTorch import time or memory requirements. `/health` remains liveness;
`/ready` retains its JSON contract (including HTTP 200 while warming), so a Render
HTTP health check alone does not gate traffic on model readiness. Existing frontend
readiness polling continues to protect query/evaluation actions during that short
initialization period. No model or cache semantics are changed.
