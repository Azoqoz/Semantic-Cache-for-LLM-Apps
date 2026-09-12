"""Run locally: python -m uvicorn src.api:app --host 127.0.0.1 --port 8000."""
from __future__ import annotations

from contextlib import asynccontextmanager
from threading import Event, Lock, Thread
from typing import Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from src.application import ApplicationError, CacheApplication, QueryCommand, create_application
from src.config import Settings, settings as default_settings


class QueryBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1)
    provider: Literal["Demo", "OpenAI", "Claude", "Gemini", "Ollama"] = "Demo"
    model: str | None = Field(default=None, min_length=1)
    threshold: float | None = Field(default=None, ge=0, le=1)
    ttl_hours: int | None = None
    isolate_by_model: bool = True


class EvaluationBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    threshold: float | None = Field(default=None, ge=0, le=1)


class QueryResponse(BaseModel):
    question: str
    answer: str
    cache_hit: bool
    hit_type: Literal["exact", "semantic", "miss"]
    similarity: float | None
    matched_question: str | None
    threshold: float
    ttl_hours: int
    isolate_by_model: bool
    provider: str
    model: str
    latency_ms: float
    estimated_cost_usd: float | None
    cost_basis: str | None
    cost_kind: Literal["avoided", "incurred"]


def create_app(application: CacheApplication | None = None, settings: Settings | None = None) -> FastAPI:
    config = application.settings if application is not None else (settings or default_settings)
    initialization_lock = Lock()
    state = "loading"
    started = False
    complete = Event()

    def warm_engine() -> None:
        nonlocal application, state
        try:
            service = application if application is not None else create_application(config)
            service.embeddings.warm_up()
            with initialization_lock:
                application = service
                state = "ready"
        except Exception:
            # Never log provider/model exceptions or expose their contents.
            with initialization_lock:
                state = "error"
        finally:
            complete.set()

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        nonlocal started
        with initialization_lock:
            if not started:
                started = True
                Thread(target=warm_engine, name="semantic-engine-warmup", daemon=True).start()
        # No inference work is awaited before ASGI startup completes / port bind.
        yield

    api = FastAPI(title="Semantic Cache API", version="1.0.0", lifespan=lifespan)
    api.state.warmup_complete = complete

    def get_application() -> CacheApplication:
        with initialization_lock:
            if state == "loading":
                raise ApplicationError("service_warming")
            if state == "error":
                raise ApplicationError("model_initialization_failed")
            return application

    @api.exception_handler(ApplicationError)
    async def application_error(request: Request, exc: ApplicationError):
        status = {"invalid_request": 422, "provider_configuration": 400, "operation_failed": 503,
                  "service_warming": 503, "model_initialization_failed": 503}[exc.code]
        return JSONResponse(status_code=status, content={"error": {"code": exc.code, "message": str(exc)}})

    @api.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Validation details include input values; do not echo bodies or keys.
        return JSONResponse(status_code=422, content={"error": {
            "code": "invalid_request", "message": ApplicationError.MESSAGES["invalid_request"]}})

    @api.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": {
            "code": "http_error", "message": "The requested route or method is unavailable."}})

    @api.middleware("http")
    async def safe_unexpected_errors(request: Request, call_next):
        # Catch before the server can log a traceback containing upstream secrets.
        try:
            return await call_next(request)
        except Exception:
            return JSONResponse(status_code=500, content={"error": {
                "code": "internal_error", "message": "An internal error occurred."}})

    @api.get("/health")
    def health():
        return {"status": "ok", "check": "liveness"}

    @api.get("/ready")
    def ready():
        with initialization_lock:
            if state == "error":
                return {"status": "error", "message": ApplicationError.MESSAGES["model_initialization_failed"]}
            return {"status": "ready" if state == "ready" else "warming"}

    @api.get("/capabilities")
    def capabilities():
        # No database or embedding initialization needed for capabilities.
        return CacheApplication.describe_capabilities(config)

    @api.post("/query", response_model=QueryResponse)
    def query(body: QueryBody, service: CacheApplication = Depends(get_application)):
        return service.query(QueryCommand(**body.model_dump()))

    @api.get("/cache")
    def cache(limit: int = Query(default=100, ge=1, le=1000),
              service: CacheApplication = Depends(get_application)):
        return service.cache_snapshot(limit)

    @api.delete("/cache")
    def clear_cache(service: CacheApplication = Depends(get_application)):
        return service.clear_cache()

    @api.post("/evaluation")
    def evaluation(body: EvaluationBody, service: CacheApplication = Depends(get_application)):
        return service.evaluate(body.threshold)

    return api


app = create_app()
