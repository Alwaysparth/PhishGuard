"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/models/schemas.py                   ║
║         Request & Response Models — Pydantic v2              ║
╚══════════════════════════════════════════════════════════════╝

Naming convention
─────────────────
  *Request   — inbound data (validated on arrival)
  *Response  — outbound data (serialised before sending)
  *Item      — internal / nested object (not a top-level schema)

All response models use `model_config = ConfigDict(from_attributes=True)`
so SQLAlchemy / sqlite3.Row objects can be passed directly to
`.model_validate()`.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
    ConfigDict,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class ScanMode(str, Enum):
    """
    viewer  — Web dashboard mode.
              Run the full analysis pipeline, but never enforce blocking.
              action is always returned as "none".

    protect — Browser extension mode.
              Enforce block / warn / allow actions based on risk score.
    """
    VIEWER  = "viewer"
    PROTECT = "protect"


class ThreatStatus(str, Enum):
    """Final classification of the scanned URL."""
    SAFE        = "safe"
    SUSPICIOUS  = "suspicious"
    PHISHING    = "phishing"


class ThreatAction(str, Enum):
    """
    Recommended action for the caller:

    allow  — URL is safe; proceed normally.
    warn   — URL is suspicious; show a warning but allow navigation.
    block  — URL is phishing; prevent navigation and show a blocked page.
    none   — No enforcement (viewer mode or informational response).
    """
    ALLOW = "allow"
    WARN  = "warn"
    BLOCK = "block"
    NONE  = "none"


class ListType(str, Enum):
    WHITELIST = "whitelist"
    BLACKLIST = "blacklist"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHECK-URL  (core detection endpoint)
# ─────────────────────────────────────────────────────────────────────────────

class CheckURLRequest(BaseModel):
    """
    POST /check-url  |  POST /api/v1/check-url

    The single request schema shared by the web dashboard and the
    browser extension.  The `mode` field drives all downstream logic.

    Example payload (web dashboard):
        { "url": "https://paypal-secure.info/login", "mode": "viewer" }

    Example payload (browser extension):
        { "url": "https://paypal-secure.info/login", "mode": "protect" }
    """

    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="The URL to analyse.  Must start with http:// or https://.",
        examples=["https://www.google.com", "http://paypal-secure-login.tk/verify"],
        json_schema_extra={"example": "https://example.com"},
    )

    mode: ScanMode = Field(
        default=ScanMode.VIEWER,
        description=(
            "'viewer' — analysis only, no enforcement (web dashboard). "
            "'protect' — enforce block/warn/allow (browser extension)."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("url", mode="before")
    @classmethod
    def normalise_url(cls, v: str) -> str:
        """Strip whitespace; prepend http:// if no scheme is present."""
        v = v.strip()
        if v and not v.startswith(("http://", "https://")):
            v = "http://" + v
        return v

    @field_validator("url")
    @classmethod
    def validate_url_structure(cls, v: str) -> str:
        """Reject obviously malformed values."""
        from urllib.parse import urlparse
        p = urlparse(v)
        if not p.netloc:
            raise ValueError("URL must contain a valid domain / host.")
        return v


class CheckURLResponse(BaseModel):
    """
    Response from POST /check-url.

    risk_score : 0–100.
                 0 = definitely safe.
                 100 = high-confidence phishing.

    status     : safe | suspicious | phishing

    action     : What the caller should do.
                 Viewer mode always returns "none".
                 Protect mode returns allow / warn / block.

    reasons    : Human-readable list of detection signals that contributed
                 to the risk score.  Empty list for clean URLs.

    Example (phishing detected):
        {
          "risk_score": 94.0,
          "status": "phishing",
          "action": "block",
          "reasons": [
            "Brand name in URL but not in registered domain",
            "No HTTPS — connection is unencrypted",
            "Suspicious top-level domain (high phishing rate)"
          ]
        }
    """

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Phishing risk score from 0 (safe) to 100 (phishing).",
    )
    status: ThreatStatus = Field(
        ...,
        description="Final threat classification.",
    )
    action: ThreatAction = Field(
        ...,
        description="Recommended action for the client.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Detection signals that contributed to the risk score.",
    )

    model_config = ConfigDict(use_enum_values=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticsTotals(BaseModel):
    """Aggregate counts across all scans ever performed."""
    total_scans:      int = Field(0, description="Total URLs ever scanned.")
    safe_count:       int = Field(0, description="URLs classified as safe.")
    suspicious_count: int = Field(0, description="URLs classified as suspicious.")
    phishing_count:   int = Field(0, description="URLs classified as phishing.")
    web_scans:        int = Field(0, description="Scans originating from the web dashboard.")
    extension_scans:  int = Field(0, description="Scans originating from the browser extension.")

    model_config = ConfigDict(from_attributes=True)


class DailyAnalyticsItem(BaseModel):
    """One row of per-day breakdown data (for the line chart)."""
    date:       str = Field(..., description="Calendar date  YYYY-MM-DD.")
    total:      int = Field(0)
    phishing:   int = Field(0)
    safe:       int = Field(0)
    suspicious: int = Field(0)

    model_config = ConfigDict(from_attributes=True)


class RecentScanItem(BaseModel):
    """Single row in the 'Recent Scans' dashboard table."""
    url:        str   = Field(..., description="Full URL that was scanned.")
    domain:     str   = Field(..., description="Registered domain extracted from URL.")
    risk_score: float = Field(..., ge=0, le=100)
    status:     str   = Field(...)
    action:     str   = Field(...)
    source:     str   = Field(..., description="'web' or 'extension'.")
    created_at: str   = Field(..., description="ISO-8601 timestamp.")

    model_config = ConfigDict(from_attributes=True)


class AnalyticsSummaryResponse(BaseModel):
    """
    GET /api/v1/analytics/summary

    Returned by the analytics endpoint consumed by the dashboard.
    """
    totals:              AnalyticsTotals        = Field(default_factory=AnalyticsTotals)
    daily:               list[DailyAnalyticsItem] = Field(default_factory=list)
    recent_scans:        list[RecentScanItem]     = Field(default_factory=list)
    blacklisted_domains: int                      = Field(0)

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DOMAIN DATABASE
# ─────────────────────────────────────────────────────────────────────────────

class DomainItem(BaseModel):
    """
    A single domain entry from the whitelist / blacklist database.
    Returned by GET /api/v1/domains and POST /api/v1/domains.
    """
    id:         Optional[int]  = Field(None, description="Auto-increment primary key.")
    domain:     str            = Field(..., min_length=2, max_length=253)
    list_type:  ListType       = Field(..., description="'whitelist' or 'blacklist'.")
    risk_score: float          = Field(0.0, ge=0, le=100)
    status:     ThreatStatus   = Field(ThreatStatus.SAFE)
    reasons:    list[str]      = Field(default_factory=list)
    scan_count: int            = Field(1, ge=0)
    created_at: Optional[str]  = None
    updated_at: Optional[str]  = None

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class AddDomainRequest(BaseModel):
    """POST /api/v1/domains — manually add a domain to the whitelist or blacklist."""
    domain:    str      = Field(..., min_length=2, max_length=253)
    list_type: ListType = Field(...)
    reasons:   list[str] = Field(default_factory=list)

    @field_validator("domain", mode="before")
    @classmethod
    def clean_domain(cls, v: str) -> str:
        """Strip scheme, path, and trailing dots."""
        import tldextract
        v = v.strip().lower()
        for prefix in ("https://", "http://", "www."):
            if v.startswith(prefix):
                v = v[len(prefix):]
        v = v.split("/")[0].rstrip(".")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# 5. HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    """ML model metadata embedded in the health response."""
    ready:       bool          = False
    trained_at:  Optional[float] = None
    s1_features: int           = 0
    s2_features: int           = 0
    thresholds:  dict          = Field(default_factory=dict)


class CacheStats(BaseModel):
    entries_total:   int = 0
    entries_valid:   int = 0
    entries_expired: int = 0
    max_size:        int = 0
    ttl_seconds:     int = 0


class HealthResponse(BaseModel):
    """
    GET /health

    Used by load-balancers, Docker health checks, and monitoring tools.
    """
    status:     str        = Field("ok", description="'ok' | 'degraded' | 'error'")
    version:    str        = Field("1.0.0")
    service:    str        = Field("PhishGuard API")
    uptime_s:   float      = Field(0.0, description="Seconds since last startup.")
    model:      ModelInfo  = Field(default_factory=ModelInfo)
    cache:      CacheStats = Field(default_factory=CacheStats)
    db_path:    str        = ""


# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERIC WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Returned on 4xx / 5xx errors."""
    error:  str            = Field(..., description="Short error message.")
    detail: Optional[str]  = Field(None, description="Optional longer description.")


class SuccessResponse(BaseModel):
    """Generic success acknowledgement for write operations."""
    ok:      bool = True
    message: str  = ""
