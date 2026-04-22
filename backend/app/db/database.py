"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/db/database.py                      ║
║         Database Setup — Schema, Seeding, CRUD Helpers       ║
╚══════════════════════════════════════════════════════════════╝

Schema overview
───────────────
  domains    — whitelist / blacklist with risk metadata
  scans      — every URL check ever performed (audit log)
  analytics  — daily aggregated counters for the dashboard

All operations use the stdlib `sqlite3` module wrapped in a simple
connection-pool pattern (one writer thread, many readers via WAL mode).
Swap out for SQLAlchemy + PostgreSQL for production scale.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger("phishguard.db")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONNECTION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

# Resolve absolute DB path from config (falls back to cwd)
try:
    from app.core.config import settings
    _DB_PATH: str = settings.DB_PATH
except Exception:
    import os
    _DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "phishguard.db")

_write_lock = threading.Lock()   # Serialise writes; SQLite allows concurrent reads in WAL mode


def _make_connection(path: str = _DB_PATH) -> sqlite3.Connection:
    """Open a new SQLite connection with sensible defaults."""
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row          # Rows accessible as dicts
    conn.execute("PRAGMA journal_mode=WAL")  # WAL → better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
    return conn


@contextmanager
def get_db():
    """
    Context manager that yields a connection and auto-commits / rolls back.

    Usage::

        with get_db() as conn:
            conn.execute("SELECT …")
    """
    conn = _make_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_write_db():
    """
    Same as get_db() but also acquires the module-level write lock so that
    only one writer runs at a time (avoids 'database is locked' errors).
    """
    with _write_lock:
        with get_db() as conn:
            yield conn


def close_db() -> None:
    """Called during application shutdown (no persistent pool to drain here)."""
    logger.info("[DB] Database connections closed.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCHEMA DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- ── domains ──────────────────────────────────────────────────────────────────
-- Stores the whitelist and blacklist.
-- Auto-populated whenever a high-risk URL is confirmed phishing.
CREATE TABLE IF NOT EXISTS domains (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    domain      TEXT    UNIQUE NOT NULL,
    list_type   TEXT    NOT NULL CHECK (list_type IN ('whitelist','blacklist')),
    risk_score  REAL    NOT NULL DEFAULT 0.0,
    status      TEXT    NOT NULL DEFAULT 'safe'
                        CHECK (status IN ('safe','suspicious','phishing')),
    reasons     TEXT    NOT NULL DEFAULT '[]',   -- JSON array of strings
    scan_count  INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_domains_domain    ON domains (domain);
CREATE INDEX IF NOT EXISTS idx_domains_list_type ON domains (list_type);

-- ── scans ─────────────────────────────────────────────────────────────────────
-- Immutable audit log — every single URL check is recorded here.
CREATE TABLE IF NOT EXISTS scans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT    NOT NULL,
    domain      TEXT    NOT NULL,
    risk_score  REAL    NOT NULL DEFAULT 0.0,
    status      TEXT    NOT NULL DEFAULT 'safe'
                        CHECK (status IN ('safe','suspicious','phishing')),
    action      TEXT    NOT NULL DEFAULT 'none'
                        CHECK (action IN ('allow','warn','block','none')),
    reasons     TEXT    NOT NULL DEFAULT '[]',   -- JSON array
    mode        TEXT    NOT NULL DEFAULT 'viewer'
                        CHECK (mode IN ('viewer','protect')),
    source      TEXT    NOT NULL DEFAULT 'web'
                        CHECK (source IN ('web','extension')),
    stage       INTEGER NOT NULL DEFAULT 1,      -- 1 = stage1 final, 2 = stage2 used
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_scans_domain     ON scans (domain);
CREATE INDEX IF NOT EXISTS idx_scans_status     ON scans (status);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans (created_at);
CREATE INDEX IF NOT EXISTS idx_scans_source     ON scans (source);

-- ── analytics ─────────────────────────────────────────────────────────────────
-- One row per calendar day, updated in real-time as scans come in.
CREATE TABLE IF NOT EXISTS analytics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    UNIQUE NOT NULL,  -- YYYY-MM-DD
    total_scans      INTEGER NOT NULL DEFAULT 0,
    safe_count       INTEGER NOT NULL DEFAULT 0,
    suspicious_count INTEGER NOT NULL DEFAULT 0,
    phishing_count   INTEGER NOT NULL DEFAULT 0,
    web_scans        INTEGER NOT NULL DEFAULT 0,
    extension_scans  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_analytics_date ON analytics (date);
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEED DATA
# ─────────────────────────────────────────────────────────────────────────────

_WHITELIST_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "twitter.com",
    "instagram.com", "linkedin.com", "github.com", "microsoft.com",
    "apple.com", "amazon.com", "netflix.com", "wikipedia.org",
    "reddit.com", "stackoverflow.com", "cloudflare.com", "anthropic.com",
    "openai.com", "mozilla.org", "python.org", "fastapi.tiangolo.com",
]

_BLACKLIST_DOMAINS = [
    ("paypal-secure-login.com",       95.0, ["Known PayPal phishing domain"]),
    ("google-verify-account.net",     94.0, ["Google impersonation phishing"]),
    ("facebook-login-secure.info",    93.0, ["Facebook credential harvester"]),
    ("amazon-prize-winner.com",       96.0, ["Amazon scam / prize phishing"]),
    ("microsoft-support-alert.net",   92.0, ["Fake Microsoft support phishing"]),
    ("apple-id-verify.info",          91.0, ["Apple ID credential theft"]),
    ("bankofamerica-secure.net",      95.0, ["Bank impersonation phishing"]),
    ("chase-bank-alert.com",          93.0, ["Chase bank phishing"]),
    ("netflix-billing-update.com",    90.0, ["Netflix billing scam"]),
    ("instagram-login-verify.net",    92.0, ["Instagram credential harvester"]),
]


def _seed_domains(conn: sqlite3.Connection) -> None:
    """Insert seed whitelist / blacklist rows (IGNORE if already present)."""
    for domain in _WHITELIST_DOMAINS:
        conn.execute(
            """INSERT OR IGNORE INTO domains
               (domain, list_type, risk_score, status, reasons)
               VALUES (?, 'whitelist', 0.0, 'safe', '[]')""",
            (domain,),
        )

    for domain, score, reasons in _BLACKLIST_DOMAINS:
        conn.execute(
            """INSERT OR IGNORE INTO domains
               (domain, list_type, risk_score, status, reasons)
               VALUES (?, 'blacklist', ?, 'phishing', ?)""",
            (domain, score, json.dumps(reasons)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. PUBLIC INITIALISATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables (if they don't exist) and seed reference data.
    Called once during application startup via the lifespan hook.
    """
    logger.info(f"[DB] Initialising database at: {_DB_PATH}")
    with get_write_db() as conn:
        conn.executescript(_SCHEMA_SQL)
        _seed_domains(conn)
    logger.info("[DB] Schema created and seed data applied.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DOMAIN CRUD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_domain(domain: str) -> Optional[dict]:
    """
    Look up a domain in the whitelist/blacklist.

    Returns a dict with domain metadata, or None if not found.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM domains WHERE domain = ?", (domain,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["reasons"] = json.loads(result.get("reasons") or "[]")
    return result


def upsert_domain(
    domain: str,
    list_type: str,
    risk_score: float,
    status: str,
    reasons: list[str],
) -> None:
    """
    Insert a new domain or update an existing one.
    Used by the auto-blacklist logic when a high-risk URL is confirmed.
    """
    reasons_json = json.dumps(reasons)
    now = datetime.utcnow().isoformat() + "Z"
    with get_write_db() as conn:
        conn.execute(
            """INSERT INTO domains (domain, list_type, risk_score, status, reasons, scan_count, updated_at)
               VALUES (?, ?, ?, ?, ?, 1, ?)
               ON CONFLICT(domain) DO UPDATE SET
                   list_type  = excluded.list_type,
                   risk_score = excluded.risk_score,
                   status     = excluded.status,
                   reasons    = excluded.reasons,
                   scan_count = scan_count + 1,
                   updated_at = excluded.updated_at""",
            (domain, list_type, risk_score, status, reasons_json, now),
        )


def list_domains(list_type: Optional[str] = None, limit: int = 100) -> list[dict]:
    """
    Return all domains of a given list type, or all domains if list_type is None.
    """
    with get_db() as conn:
        if list_type:
            rows = conn.execute(
                "SELECT * FROM domains WHERE list_type = ? ORDER BY updated_at DESC LIMIT ?",
                (list_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM domains ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        r["reasons"] = json.loads(r.get("reasons") or "[]")
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. SCAN CRUD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def insert_scan(
    url: str,
    domain: str,
    risk_score: float,
    status: str,
    action: str,
    reasons: list[str],
    mode: str,
    stage: int = 1,
) -> int:
    """
    Write one scan record and update the daily analytics counter.
    Returns the new scan row ID.
    """
    source = "extension" if mode == "protect" else "web"
    reasons_json = json.dumps(reasons)
    today = date.today().isoformat()

    with get_write_db() as conn:
        cursor = conn.execute(
            """INSERT INTO scans
               (url, domain, risk_score, status, action, reasons, mode, source, stage)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (url, domain, risk_score, status, action, reasons_json, mode, source, stage),
        )
        scan_id = cursor.lastrowid

        # ── Update daily analytics counter (upsert) ──────────────────────────
        status_col   = f"{status}_count"          # safe_count / suspicious_count / phishing_count
        source_col   = f"{source}_scans"          # web_scans / extension_scans

        conn.execute(
            f"""INSERT INTO analytics (date, total_scans, {status_col}, {source_col})
                VALUES (?, 1, 1, 1)
                ON CONFLICT(date) DO UPDATE SET
                    total_scans    = total_scans + 1,
                    {status_col}   = {status_col} + 1,
                    {source_col}   = {source_col} + 1""",
            (today,),
        )

    return scan_id


def get_recent_scans(limit: int = 20) -> list[dict]:
    """Return the most recent scan records."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        r["reasons"] = json.loads(r.get("reasons") or "[]")
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANALYTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_analytics_summary() -> dict:
    """
    Return a summary dict used by the /analytics/summary endpoint.
    Includes:
      • Lifetime totals
      • Last 30 days of daily rows (for line chart)
      • 10 most recent scans (for the dashboard table)
    """
    with get_db() as conn:
        # Lifetime totals (aggregate entire scans table)
        totals_row = conn.execute(
            """SELECT
                COUNT(*)                                              AS total_scans,
                SUM(CASE WHEN status = 'safe'       THEN 1 ELSE 0 END) AS safe_count,
                SUM(CASE WHEN status = 'suspicious' THEN 1 ELSE 0 END) AS suspicious_count,
                SUM(CASE WHEN status = 'phishing'   THEN 1 ELSE 0 END) AS phishing_count,
                SUM(CASE WHEN source = 'web'        THEN 1 ELSE 0 END) AS web_scans,
                SUM(CASE WHEN source = 'extension'  THEN 1 ELSE 0 END) AS extension_scans
               FROM scans"""
        ).fetchone()

        # Daily breakdown for the past 30 days (for the line chart)
        daily_rows = conn.execute(
            """SELECT date(created_at) AS date,
                      COUNT(*)                                                AS total,
                      SUM(CASE WHEN status = 'phishing' THEN 1 ELSE 0 END)  AS phishing,
                      SUM(CASE WHEN status = 'safe'     THEN 1 ELSE 0 END)  AS safe,
                      SUM(CASE WHEN status = 'suspicious' THEN 1 ELSE 0 END) AS suspicious
               FROM scans
               GROUP BY date(created_at)
               ORDER BY date DESC
               LIMIT 30"""
        ).fetchall()

        # Recent scan rows for the dashboard table
        recent_rows = conn.execute(
            """SELECT url, domain, risk_score, status, action, source, created_at
               FROM scans ORDER BY created_at DESC LIMIT 10"""
        ).fetchall()

        # Blacklist count
        blacklist_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM domains WHERE list_type = 'blacklist'"
        ).fetchone()["cnt"]

    return {
        "totals": dict(totals_row) if totals_row else {},
        "daily": [dict(r) for r in daily_rows],
        "recent_scans": [dict(r) for r in recent_rows],
        "blacklisted_domains": blacklist_count,
    }
