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
python -m uvicorn src.api:app --host 0.0.0.0 --port $PORT --workers 1
```

The prefetch module reads `settings.embedding_model` (still
`sentence-transformers/all-MiniLM-L6-v2`) and fetches the official full-precision
`onnx/model.onnx`, tokenizer and configuration files at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Artifacts are stored under
`.model-cache/sentence-transformers--all-MiniLM-L6-v2/onnx-1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
The build validates a real encode using the same offline runtime loader.
Download, artifact, or inference errors fail the build. Only the build module
imports Hugging Face Hub; API execution never does.

The path is relative to the project root, not the shell working directory or a
build-only home cache. Do not mount a runtime disk over `.model-cache` or exclude
it from a custom deployment artifact. The directory is Git-ignored.

Runtime uses `onnxruntime` with `CPUExecutionProvider`, `tokenizers` and NumPy.
It opens explicit local filenames and contains no remote-loading fallback, even
in local development. Run prefetch once before local use as well. PyTorch and
SentenceTransformer are development-only reference dependencies. The tokenizer
retains the model's special tokens, uncased normalization and 256-token limit.
Attention-mask mean pooling and L2 normalization produce 384 float32 components.
No quantization is used. The official export requires pooling and normalization
outside ONNX, as described in the
[Sentence Transformers documentation](https://github.com/huggingface/sentence-transformers/blob/main/docs/sentence_transformer/usage/efficiency.rst).
Inference uses one CPU thread and serializes encode calls to bound activation
memory. Existing SQLite vectors remain compatible; no database reset or schema
change is required. Near an exact threshold, tiny floating-point differences can
still affect a decision; thresholds and inclusive comparisons are unchanged.

FastAPI still binds without waiting; background startup loads the cached weights
into memory once, then `/ready` becomes `ready`. Prefetch removes download time,
not all process startup or model memory requirements. `/health` remains liveness;
`/ready` retains its JSON contract (including HTTP 200 while warming), so a Render
HTTP health check alone does not gate traffic on model readiness. Existing frontend
readiness polling continues to protect query/evaluation actions during that short
initialization period. No model or cache semantics are changed.

Warm-up still runs once in the background, tests one small encode, and has a
180-second terminal timeout. Logs now show `importing_onnx_runtime`,
`onnx_runtime_imported`, `constructing_onnx_session`, `onnx_session_constructed`,
`first_test_encode` and `model_ready`, with elapsed time and Linux RSS. No request
can start initialization. Missing artifacts and timeout require an explicit
rebuild/restart rather than automatic download retries.

## Real embedding parity tests

Install `requirements-dev.txt` and run production prefetch first. Download the
pinned PyTorch reference separately (development only):

```sh
python -c "from huggingface_hub import snapshot_download; from src.onnx_embeddings import MODEL_NAME, MODEL_REVISION, MODEL_FILES; snapshot_download(MODEL_NAME, revision=MODEL_REVISION, local_dir='.model-cache/parity-reference', allow_patterns=[f for f in MODEL_FILES if not f.endswith('.onnx')] + ['model.safetensors'])"
```

Set `SENTENCE_TRANSFORMER_REFERENCE` to the absolute path of
`.model-cache/parity-reference`, then run:

```sh
python -B -m unittest tests.test_embedding_onnx_parity tests.test_startup -v
python -B -m unittest discover -v
```

Parity tests use real offline inference on the 36 evaluation pairs plus Unicode,
whitespace, and truncation cases. They check dimensions, normalization, numerical
tolerance, ranking, all evaluation thresholds, evaluation metrics, and reuse of
persisted PyTorch embeddings. A fresh-process API test blocks PyTorch,
Transformers, SentenceTransformer, Hugging Face Hub and external network access.
Without local artifacts/reference configuration these integration tests explicitly
skip; ordinary parity/unit tests require no downloads.
