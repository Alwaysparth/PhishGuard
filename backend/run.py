"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · run.py                                  ║
║         Entry Point — starts the Uvicorn server              ║
╚══════════════════════════════════════════════════════════════╝

Usage
─────
  # Development (auto-reload):
  python run.py

  # Production:
  ENV=production python run.py

  # Via Uvicorn directly:
  uvicorn app.core.config:app --host 0.0.0.0 --port 8000
"""

import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.core.config:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
