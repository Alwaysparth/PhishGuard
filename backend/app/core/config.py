"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/core/config.py                      ║
║         Application Setup — Factory, Middleware, Lifecycle   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# 1. SETTINGS  (reads from environment / .env file)
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "PhishGuard"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Centralized AI-powered phishing detection API. "
        "Serves both the web dashboard (viewer mode) and "
        "the browser extension (protect mode)."
    )
    DEBUG: bool = False
    ENV: str = "production"          # "development" | "production"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Security
    SECRET_KEY: str = "change-me-in-production-use-secrets-module"
    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[str] = ["*"]
    CORS_METHODS: list[str] = ["GET", "POST", "OPTIONS"]
    CORS_HEADERS: list[str] = ["*"]

    # Database
    DB_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "phishguard.db")
    DB_POOL_SIZE: int = 5
    DB_ECHO: bool = False

    # ML Model
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "models", "phishguard_model.pkl")
    MODEL_RELOAD_INTERVAL_HOURS: int = 24   # Auto-retrain cadence (future use)
    STAGE1_PHISHING_THRESHOLD: float = 0.90  # S1 ≥ 0.90 → phishing (skip stage 2)
    STAGE1_SAFE_THRESHOLD: float = 0.65      # S1 < 0.65 → safe    (skip stage 2)
    STAGE2_PHISHING_THRESHOLD: float = 0.80  # S2 ≥ 0.80 → phishing

    # Cache
    URL_CACHE_TTL_SECONDS: int = 600          # 10-minute result cache
    URL_CACHE_MAX_SIZE: int = 1000

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton instance used across the entire application
settings = Settings()


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def configure_logging() -> logging.Logger:
    """Configure root logger and return the app-level logger."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=settings.LOG_FORMAT,
    )
    logger = logging.getLogger("phishguard")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger


logger = configure_logging()


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIFESPAN  (startup / shutdown hooks)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup (before first request) and once on shutdown.

    Startup tasks:
      • Initialise the SQLite database (creates tables + seeds known domains)
      • Load / train the ML model into memory
      • Warm up the URL result cache

    Shutdown tasks:
      • Flush pending DB writes
      • Persist the in-memory URL cache to disk (optional)
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION} — starting up")
    logger.info(f"  Environment : {settings.ENV}")
    logger.info(f"  Debug mode  : {settings.DEBUG}")
    logger.info("=" * 60)

    # 1. Database
    from app.db.database import init_db
    init_db()
    logger.info("[DB]    Database initialised and seeded.")

    # 2. ML Model
    from app.ml.model import PhishingModel
    app.state.model = PhishingModel()
    app.state.model.load_or_train()
    logger.info("[ML]    Phishing detection model ready.")

    # 3. In-memory URL cache
    from app.core.cache import URLCache
    app.state.url_cache = URLCache(
        ttl_seconds=settings.URL_CACHE_TTL_SECONDS,
        max_size=settings.URL_CACHE_MAX_SIZE,
    )
    logger.info("[CACHE] URL result cache initialised.")

    logger.info("[UP]    PhishGuard API is ready to serve requests.")
    logger.info("=" * 60)

    yield   # ← application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("[DOWN]  PhishGuard shutting down — cleaning up resources…")
    from app.db.database import close_db
    close_db()
    logger.info("[DOWN]  Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. APPLICATION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Returns a fully configured FastAPI instance with:
      - Lifespan hooks (startup / shutdown)
      - CORS middleware
      - Trusted-host middleware
      - Custom 404 / 422 / 500 error handlers
      - All API routers mounted under /api/v1
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/docs" if settings.DEBUG else None,   # Hide Swagger in production
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    _register_middleware(app)
    _register_routers(app)
    _register_error_handlers(app)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# 5. MIDDLEWARE REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

def _register_middleware(app: FastAPI) -> None:
    """Attach all middleware layers to the application."""

    # CORS — allow the web dashboard and browser extension to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )

    # Trusted-host protection (only relevant when not behind a reverse proxy)
    if settings.ALLOWED_HOSTS != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    logger.debug("[MIDDLEWARE] CORS and TrustedHost middleware registered.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ROUTER REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

def _register_routers(app: FastAPI) -> None:
    """Mount all API routers."""
    from app.api.endpoints import router as api_router
    from app.api.endpoints import check_url_endpoint, health_check 

    app.include_router(api_router, prefix="/api/v1")

    # Backward-compatible top-level shortcut: POST /check-url
    # (the browser extension and legacy web app call this directly)
    from app.api.endpoints import check_url_endpoint
    app.add_api_route(
        "/check-url",
        check_url_endpoint,
        methods=["POST"],
        summary="Check URL (top-level shortcut)",
        tags=["Detection"],
    )
    app.add_api_route(
        "/health",
        health_check,
        methods=["GET"], 
        tags=["System"]
    )

    logger.debug("[ROUTER] API routers mounted.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def _register_error_handlers(app: FastAPI) -> None:
    """Register custom JSON error responses for common HTTP errors."""

    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        return JSONResponse(
            status_code=404,
            content={"error": "Not found", "path": str(request.url.path)},
        )

    @app.exception_handler(422)
    async def validation_error_handler(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "detail": exc.errors() if hasattr(exc, "errors") else str(exc),
            },
        )

    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        logger.error(f"Internal server error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error. Please try again later."},
        )

    logger.debug("[ERRORS] Custom error handlers registered.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. IN-MEMORY URL CACHE  (lightweight; no Redis dependency)
# ─────────────────────────────────────────────────────────────────────────────

# app/core/cache.py is imported above — defined here to keep the module count
# compact; move to its own file if the cache logic grows.

# ─────────────────────────────────────────────────────────────────────────────
# 9. ENTRY POINT  (python -m app.core.config  OR  uvicorn app.core.config:app)
# ─────────────────────────────────────────────────────────────────────────────

# Top-level `app` object for Uvicorn:  uvicorn app.core.config:app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.core.config:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
