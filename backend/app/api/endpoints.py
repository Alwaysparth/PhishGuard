"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/api/endpoints.py                    ║
║         API Endpoints — All Routes                           ║
╚══════════════════════════════════════════════════════════════╝

Route map
─────────
  POST  /check-url                (top-level shortcut)
  POST  /api/v1/check-url         Detection pipeline (main endpoint)
  GET   /api/v1/analytics/summary Dashboard statistics
  GET   /api/v1/domains           List whitelist / blacklist
  POST  /api/v1/domains           Manually add a domain
  DELETE /api/v1/domains/{domain} Remove a domain
  GET   /health                   Health check (load-balancer / Docker)

Flow for POST /check-url
────────────────────────
  1. Validate + normalise URL
  2. Extract registered domain
  3. Hit DB whitelist / blacklist cache
       └─ HIT  → return stored result
       └─ MISS → Stage 1 feature extraction + ML prediction
                    S1 ≥ 0.90  → phishing  (record + auto-blacklist)
                    S1 < 0.65  → safe      (record)
                    else       → Stage 2 deep analysis
                                   S2 ≥ 0.80  → phishing
                                   S2 < 0.80  → suspicious
  4. Apply mode-based action (viewer → "none", protect → enforce)
  5. Persist scan record + update analytics
  6. Return CheckURLResponse
"""

import logging
import time
from typing import Optional

import tldextract
from fastapi import APIRouter, Request, HTTPException, Query, Depends

from app.models.schemas import (
    CheckURLRequest,
    CheckURLResponse,
    ThreatStatus,
    ThreatAction,
    ScanMode,
    AnalyticsSummaryResponse,
    AnalyticsTotals,
    DailyAnalyticsItem,
    RecentScanItem,
    DomainItem,
    AddDomainRequest,
    HealthResponse,
    ModelInfo,
    CacheStats,
    SuccessResponse,
    ErrorResponse,
    ListType,
)
from app.db.database import (
    get_domain,
    upsert_domain,
    insert_scan,
    get_recent_scans,
    list_domains,
    get_analytics_summary,
)
from app.ml.features import extract_stage1, extract_stage2

logger = logging.getLogger("phishguard.api")

router = APIRouter()

# ─── Startup timestamp (used in /health uptime) ───────────────────────────────
_START_TIME = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_model(request: Request):
    """Retrieve the shared PhishingModel from app state."""
    model = getattr(request.app.state, "model", None)
    if model is None or not model.is_ready:
        raise HTTPException(
            status_code=503,
            detail="ML model is not yet initialised. Please retry shortly.",
        )
    return model


def _get_cache(request: Request):
    """Retrieve the shared URLCache from app state."""
    return getattr(request.app.state, "url_cache", None)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _determine_action(status: ThreatStatus, mode: ScanMode) -> ThreatAction:
    """
    Map (status, mode) → action.

    Viewer mode always returns NONE — the web dashboard shows results
    without blocking or redirecting the user.
    """
    if mode == ScanMode.VIEWER:
        return ThreatAction.NONE

    # Protect mode (browser extension)
    mapping = {
        ThreatStatus.PHISHING:   ThreatAction.BLOCK,
        ThreatStatus.SUSPICIOUS: ThreatAction.WARN,
        ThreatStatus.SAFE:       ThreatAction.ALLOW,
    }
    return mapping.get(status, ThreatAction.NONE)


def _score_to_status(risk_score: float) -> ThreatStatus:
    if risk_score >= 90:
        return ThreatStatus.PHISHING
    if risk_score >= 65:
        return ThreatStatus.SUSPICIOUS
    return ThreatStatus.SAFE


def _run_detection_pipeline(url: str, model, cache) -> CheckURLResponse:
    """
    Core two-stage detection pipeline.

    Parameters
    ──────────
    url    : Normalised URL string
    model  : PhishingModel instance (from app.state)
    cache  : URLCache instance (from app.state) — may be None

    Returns
    ───────
    (risk_score, status, reasons, stage_used)
    as a plain tuple — action is added by the endpoint.
    """

    # ── 1. URL cache check ────────────────────────────────────────────────────
    if cache:
        cached = cache.get(url)
        if cached:
            logger.debug(f"[CACHE HIT] {url}")
            return cached  # cached value is already a (risk_score, status, reasons, stage) tuple

    # ── 2. Database whitelist / blacklist check ───────────────────────────────
    ext    = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    db_entry = get_domain(domain)
    if db_entry:
        logger.debug(f"[DB HIT] {domain} → {db_entry['list_type']}")
        risk_score = float(db_entry["risk_score"])
        reasons    = db_entry.get("reasons") or []
        if db_entry["list_type"] == "whitelist":
            risk_score = 0.0
            reasons    = ["Domain is on the trusted whitelist"]
        elif not reasons:
            reasons = ["Domain is on the known phishing blacklist"]
        result = (risk_score, reasons, 0)   # stage=0 means DB hit
        if cache:
            cache.set(url, result)
        return result

    # ── 3. Stage 1 — URL-level ML prediction ─────────────────────────────────
    s1_features             = extract_stage1(url)
    s1_prob, s1_reasons     = model.predict_stage1(url)
    s1_score                = round(s1_prob * 100, 1)

    logger.debug(f"[S1] {url}  →  {s1_score:.1f}%")

    if s1_score >= 90.0:
        # High-confidence phishing — skip Stage 2
        result = (s1_score, s1_reasons or ["High-confidence phishing indicators"], 1)
        if cache:
            cache.set(url, result)
        return result

    if s1_score < 65.0:
        # Clean URL — skip Stage 2
        result = (
            s1_score,
            s1_reasons if s1_reasons else ["No significant threat indicators detected"],
            1,
        )
        if cache:
            cache.set(url, result)
        return result

    # ── 4. Stage 2 — deep / behavioral analysis ────────────────────────────────
    s2_prob, s2_reasons = model.predict_stage2(url, s1_features)
    s2_score            = round(s2_prob * 100, 1)

    logger.debug(f"[S2] {url}  →  {s2_score:.1f}%")

    combined_reasons = s2_reasons if s2_reasons else (
        ["Deep analysis identified suspicious behavioral patterns"]
        if s2_score >= 65
        else ["URL shows moderate risk indicators — proceed with caution"]
    )

    result = (s2_score, combined_reasons, 2)
    if cache:
        cache.set(url, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — POST /check-url  (main detection)
# ─────────────────────────────────────────────────────────────────────────────

async def check_url_endpoint(
    payload: CheckURLRequest,
    request: Request,
    model=Depends(_get_model),
    cache=Depends(_get_cache),
) -> CheckURLResponse:
    """
    **Primary phishing detection endpoint.**

    Accepts a URL and a mode, runs the two-stage ML pipeline, and returns
    a structured threat assessment.

    - **viewer mode** (web dashboard): full analysis, action = `none`
    - **protect mode** (browser extension): full analysis + enforce action

    The response `action` field tells the client what to do:
    - `allow`  — URL is safe
    - `warn`   — show a caution banner (suspicious)
    - `block`  — redirect to the warning page (phishing)
    - `none`   — informational only (viewer mode)
    """
    url  = payload.url
    mode = payload.mode

    t0 = time.perf_counter()

    # Run detection
    risk_score, reasons, stage = _run_detection_pipeline(url, model, cache)

    status = _score_to_status(risk_score)
    action = _determine_action(status, mode)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[SCAN] url={url!r}  score={risk_score:.1f}  "
        f"status={status.value}  action={action.value}  "
        f"mode={mode.value}  stage={stage}  time={elapsed*1000:.1f}ms"
    )

    # Persist scan + auto-blacklist high-risk domains
    ext    = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain

    if stage > 0:    # Don't double-write for DB-hit results
        insert_scan(
            url=url,
            domain=domain,
            risk_score=risk_score,
            status=status.value,
            action=action.value,
            reasons=reasons,
            mode=mode.value,
            stage=stage,
        )

        if risk_score >= 90:
            upsert_domain(
                domain=domain,
                list_type="blacklist",
                risk_score=risk_score,
                status="phishing",
                reasons=reasons[:3],    # Store top 3 reasons only
            )
            # Invalidate cache for this domain
            if cache:
                cache.delete(url)

    return CheckURLResponse(
        risk_score=risk_score,
        status=status,
        action=action,
        reasons=reasons,
    )


# Register on the router too (under /api/v1 prefix)
router.add_api_route(
    "/check-url",
    check_url_endpoint,
    methods=["POST"],
    response_model=CheckURLResponse,
    summary="Analyse a URL for phishing threats",
    description=(
        "Two-stage ML phishing detection. "
        "Pass `mode='viewer'` for the web dashboard (no blocking enforced). "
        "Pass `mode='protect'` for the browser extension (block/warn enforced)."
    ),
    tags=["Detection"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — GET /api/v1/analytics/summary
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    summary="Dashboard analytics summary",
    tags=["Analytics"],
)
async def analytics_summary() -> AnalyticsSummaryResponse:
    """
    Return aggregate statistics for the web dashboard:

    - **totals**: lifetime scan counts by status and source
    - **daily**: last 30 days of per-day breakdown (for the line chart)
    - **recent_scans**: 10 most recent scan records (for the table)
    - **blacklisted_domains**: count of auto-blacklisted domains
    """
    raw = get_analytics_summary()

    totals = AnalyticsTotals(**raw.get("totals", {}))

    daily = [
        DailyAnalyticsItem(**row)
        for row in raw.get("daily", [])
    ]

    recent = [
        RecentScanItem(**row)
        for row in raw.get("recent_scans", [])
    ]

    return AnalyticsSummaryResponse(
        totals=totals,
        daily=daily,
        recent_scans=recent,
        blacklisted_domains=raw.get("blacklisted_domains", 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — GET /api/v1/domains
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/domains",
    response_model=list[DomainItem],
    summary="List whitelist / blacklist domains",
    tags=["Domain Management"],
)
async def list_all_domains(
    list_type: Optional[str] = Query(
        None,
        description="Filter by 'whitelist' or 'blacklist'.  Omit for all.",
        pattern="^(whitelist|blacklist)$",
    ),
    limit: int = Query(100, ge=1, le=500, description="Max rows to return."),
) -> list[DomainItem]:
    """
    Return domains from the internal whitelist / blacklist database.

    Query parameters:
    - `list_type` — filter by `whitelist` or `blacklist`
    - `limit` — cap the number of results (default 100)
    """
    rows = list_domains(list_type=list_type, limit=limit)
    return [DomainItem(**row) for row in rows]


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — POST /api/v1/domains
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/domains",
    response_model=SuccessResponse,
    status_code=201,
    summary="Manually add a domain to whitelist / blacklist",
    tags=["Domain Management"],
)
async def add_domain(
    payload: AddDomainRequest,
    cache=Depends(_get_cache),
) -> SuccessResponse:
    """
    Manually whitelist or blacklist a domain.

    The domain is stored in the DB and any existing cache entry for URLs
    on that domain is invalidated.
    """
    risk_score = 0.0 if payload.list_type == ListType.WHITELIST else 95.0
    status     = "safe"  if payload.list_type == ListType.WHITELIST else "phishing"

    upsert_domain(
        domain=payload.domain,
        list_type=payload.list_type.value,
        risk_score=risk_score,
        status=status,
        reasons=payload.reasons,
    )

    if cache:
        # Invalidate any cached results for this domain
        # (URLCache doesn't support prefix-based deletion, so we clear the whole cache
        #  only if it's small; otherwise we rely on TTL expiry)
        if cache.size < 50:
            cache.clear()

    logger.info(f"[DOMAIN] Added {payload.domain} to {payload.list_type.value}")

    return SuccessResponse(
        ok=True,
        message=f"Domain '{payload.domain}' added to {payload.list_type.value}.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 — DELETE /api/v1/domains/{domain}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/domains/{domain}",
    response_model=SuccessResponse,
    summary="Remove a domain from the database",
    tags=["Domain Management"],
)
async def remove_domain(domain: str) -> SuccessResponse:
    """
    Remove a domain entry from the whitelist / blacklist.

    The next scan for that domain will go through the full ML pipeline.
    """
    from app.db.database import get_write_db
    with get_write_db() as conn:
        conn.execute("DELETE FROM domains WHERE domain = ?", (domain,))
    logger.info(f"[DOMAIN] Removed {domain} from DB")
    return SuccessResponse(ok=True, message=f"Domain '{domain}' removed.")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 6 — GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["System"],
)
async def health_check(request: Request) -> HealthResponse:
    """
    Lightweight health-check endpoint for load balancers and Docker HEALTHCHECK.

    Returns:
    - `status`   — "ok" | "degraded" | "error"
    - `uptime_s` — seconds since last (re)start
    - `model`    — ML model readiness and metadata
    - `cache`    — URL cache statistics
    """
    model = getattr(request.app.state, "model", None)
    cache = getattr(request.app.state, "url_cache", None)

    model_info = ModelInfo(**model.info()) if model else ModelInfo()
    cache_stats = CacheStats(**cache.stats()) if cache else CacheStats()

    overall = "ok" if (model_info.ready) else "degraded"

    try:
        from app.core.config import settings
        db_path = settings.DB_PATH
    except Exception:
        db_path = "unknown"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        service="PhishGuard API",
        uptime_s=round(time.monotonic() - _START_TIME, 2),
        model=model_info,
        cache=cache_stats,
        db_path=db_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL SHORTCUT  (for backward compat with extension + frontend)
# ─────────────────────────────────────────────────────────────────────────────
# The `check_url_endpoint` function is imported by app/core/config.py and
# mounted directly at POST /check-url (no /api/v1 prefix) so that callers
# that hard-code the old URL continue to work without modification.
