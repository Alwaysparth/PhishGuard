"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/core/config.py                      ║
║         Application Setup — Render-Ready Production Config   ║
╚══════════════════════════════════════════════════════════════╝
"""
 
import os
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
 
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    from pydantic import BaseSettings
    HAS_PYDANTIC_SETTINGS = False
 
# ─────────────────────────────────────────────────────────────────────────────
# RENDER PERSISTENT DISK
# ─────────────────────────────────────────────────────────────────────────────
# On Render free tier, the filesystem resets on every deploy EXCEPT for
# a mounted persistent disk (available on paid plans).
# On the free tier, models are retrained on every cold start (~30s).
# To avoid retraining, commit your trained .pkl files to the repo
# or use Render's Disk feature.
#
# Default paths:
#   /opt/render/project/src/  ← your repo root on Render
#   We store DB and models relative to the project root.
 
_RENDER_ROOT = os.environ.get(
    "RENDER_PROJECT_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_DATA_DIR   = os.path.join(_RENDER_ROOT, "data")
_MODELS_DIR = os.path.join(_RENDER_ROOT, "models")
 
os.makedirs(_DATA_DIR,   exist_ok=True)
os.makedirs(_MODELS_DIR, exist_ok=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
 
class Settings(BaseSettings):
    # Application
    APP_NAME:        str  = "PhishGuard"
    APP_VERSION:     str  = "1.0.0"
    APP_DESCRIPTION: str  = (
        "Centralized AI-powered phishing detection API. "
        "Serves both the web dashboard (viewer mode) and "
        "the browser extension (protect mode)."
    )
    DEBUG: bool = False
    ENV:   str  = "production"
 
    # Server
    HOST:    str = "0.0.0.0"
    PORT:    int = 8000
    WORKERS: int = 1
 
    # Security / CORS
    # ─────────────────────────────────────────────────────────────────────────
    # IMPORTANT: After deploying your frontend, add its URL here.
    # Set via Render environment variable:
    #   CORS_ORIGINS=["https://your-app.vercel.app","http://localhost:5500"]
    # ─────────────────────────────────────────────────────────────────────────
    SECRET_KEY:    str       = os.environ.get("SECRET_KEY", "change-me-in-production")
    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS:  list[str] = [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "*",   # Remove this and add your real frontend URL in production
    ]
    CORS_METHODS: list[str] = ["GET", "POST", "OPTIONS", "DELETE"]
    CORS_HEADERS: list[str] = ["*"]
 
    # Database — use persistent disk path on Render
    DB_PATH: str = os.path.join(_DATA_DIR, "phishguard.db")
 
    # ML Model — stored in models/ dir
    MODEL_PATH: str = os.path.join(_MODELS_DIR, "phishguard_model.pkl")
 
    # Decision thresholds
    STAGE1_PHISHING_THRESHOLD: float = 0.90
    STAGE1_SAFE_THRESHOLD:     float = 0.65
    STAGE2_PHISHING_THRESHOLD: float = 0.80
 
    # Cache
    URL_CACHE_TTL_SECONDS: int = 600
    URL_CACHE_MAX_SIZE:    int = 1000
 
    # Logging
    LOG_LEVEL:  str = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
 
    if HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=True,
            extra="ignore",
        )
 
 
settings = Settings()
 
# Track startup time for uptime reporting
_START_TIME = time.monotonic()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGGING
# ─────────────────────────────────────────────────────────────────────────────
 
def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=settings.LOG_FORMAT,
    )
    logger = logging.getLogger("phishguard")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger
 
 
logger = configure_logging()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION} — starting")
    logger.info(f"  Environment : {settings.ENV}")
    logger.info(f"  DB path     : {settings.DB_PATH}")
    logger.info(f"  Model path  : {settings.MODEL_PATH}")
    logger.info("=" * 60)
 
    # 1. Database
    from app.db.database import init_db
    init_db()
    logger.info("[DB]    Database initialised.")
 
    # 2. ML Model
    from app.ml.model import PhishingModel
    app.state.model = PhishingModel()
    app.state.model.load_or_train()
    logger.info("[ML]    Model ready.")
 
    # 3. URL Cache
    from app.core.cache import URLCache
    app.state.url_cache = URLCache(
        ttl_seconds=settings.URL_CACHE_TTL_SECONDS,
        max_size=settings.URL_CACHE_MAX_SIZE,
    )
    logger.info("[CACHE] URL cache initialised.")
    logger.info("[UP]    PhishGuard API is ready.")
 
    yield
 
    logger.info("[DOWN]  Shutting down…")
    from app.db.database import close_db
    close_db()
    logger.info("[DOWN]  Done.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────
 
def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/docs",       # Always enable — useful for testing on Render
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    _register_middleware(application)
    _register_routers(application)
    _register_error_handlers(application)
    return application
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────
 
def _register_middleware(application: FastAPI) -> None:
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. ROUTERS
# ─────────────────────────────────────────────────────────────────────────────
 
def _register_routers(application: FastAPI) -> None:
    from app.api.endpoints import router as api_router
    from app.api.endpoints import check_url_endpoint, health_check
 
    application.include_router(api_router, prefix="/api/v1")
 
    # Top-level shortcuts (used by frontend + extension directly)
    application.add_api_route(
        "/check-url",
        check_url_endpoint,
        methods=["POST"],
        summary="Check URL — top-level shortcut",
        tags=["Detection"],
    )
    application.add_api_route(
        "/health",
        health_check,
        methods=["GET"],
        summary="Health check",
        tags=["System"],
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────
 
def _register_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(404)
    async def not_found(request, exc):
        return JSONResponse(
            status_code=404,
            content={"error": "Not found", "path": str(request.url.path)},
        )
 
    @application.exception_handler(422)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "detail": exc.errors() if hasattr(exc, "errors") else str(exc),
            },
        )
 
    @application.exception_handler(500)
    async def server_error(request, exc):
        logger.error(f"Internal error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error. Please try again."},
        )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. APP INSTANCE  (Uvicorn entry point)
# ─────────────────────────────────────────────────────────────────────────────
 
app = create_app()
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.core.config:app",
        host=settings.HOST,
        port=int(os.environ.get("PORT", settings.PORT)),
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
 