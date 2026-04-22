"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/ml/features.py                      ║
║         Feature Extraction — Stage 1 (URL) & Stage 2 (Deep) ║
╚══════════════════════════════════════════════════════════════╝

Stage 1 features  (fast — extracted from the URL string alone)
─────────────────────────────────────────────────────────────
  URL structure    : length, special-char counts, depth, port, IP usage
  Domain           : entropy, digit ratio, hyphen abuse, subdomain depth
  Security         : HTTPS presence, known URL shorteners
  Brand signals    : brand name in URL but NOT in registered domain
  Keywords         : login, verify, secure, update …

Stage 2 features  (deep — simulates JS/HTML analysis; extend with real HTTP
  fetching in production)
────────────────────────────────────────────────────────────
  Form analysis    : login form detected, external form action
  Obfuscation      : encoded chars, base64 in query, double-slash path
  Tricks           : homograph (unicode), fake-HTTPS prefix, punycode
  Redirect chains  : multiple http:// occurrences in URL
"""

import math
import re
from urllib.parse import urlparse, unquote
from dataclasses import dataclass, field, asdict
from typing import Optional

import tldextract


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS: list[str] = [
    "login", "signin", "sign-in", "verify", "verification",
    "secure", "account", "update", "confirm", "banking",
    "paypal", "amazon", "apple", "microsoft", "google",
    "facebook", "password", "credential", "security", "alert",
    "suspended", "validate", "recover", "unlock", "billing",
    "invoice", "payment", "helpdesk", "support", "2fa",
    "auth", "authorize", "reset", "webscr",
]

BRAND_NAMES: list[str] = [
    "paypal", "google", "facebook", "amazon", "apple", "microsoft",
    "netflix", "instagram", "twitter", "linkedin", "ebay", "chase",
    "wellsfargo", "bankofamerica", "citibank", "hsbc", "barclays",
    "dropbox", "icloud", "onedrive", "outlook", "yahoo", "whatsapp",
    "telegram", "coinbase", "binance", "blockchain",
]

URL_SHORTENERS: set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "buff.ly", "rebrand.ly", "shorturl.at", "is.gd", "v.gd",
    "tiny.cc", "clck.ru", "cutt.ly", "rb.gy",
}

SUSPICIOUS_TLDS: set[str] = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".info", ".biz", ".club", ".work", ".click", ".link",
    ".online", ".site", ".tech", ".pw", ".cc",
}

SUSPICIOUS_FILE_EXTS: set[str] = {
    ".exe", ".php", ".js", ".vbs", ".bat", ".cmd", ".scr",
    ".pif", ".msi", ".dll", ".jar", ".apk",
}

_IP_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage1Features:
    """
    All URL-level and domain-level features used for the first-pass ML
    prediction.  Values are numeric (int or float) so they can be passed
    directly to sklearn / XGBoost as a feature vector.
    """
    # ── URL length ────────────────────────────────────────────────────────────
    url_length: int = 0
    domain_length: int = 0
    subdomain_length: int = 0
    path_length: int = 0
    query_length: int = 0

    # ── Character counts ─────────────────────────────────────────────────────
    dot_count: int = 0
    slash_count: int = 0
    hyphen_count: int = 0
    at_count: int = 0
    question_count: int = 0
    equals_count: int = 0
    underscore_count: int = 0
    tilde_count: int = 0
    percent_count: int = 0
    amp_count: int = 0
    hash_count: int = 0

    # ── Structural flags ─────────────────────────────────────────────────────
    has_https: int = 0           # 1 = HTTPS present
    has_port: int = 0            # 1 = non-default port in URL
    has_ip: int = 0              # 1 = IP address used instead of domain name
    is_shortener: int = 0        # 1 = well-known URL shortener
    double_slash_path: int = 0   # 1 = // in path (redirect trick)
    has_suspicious_ext: int = 0  # 1 = .exe, .php, etc. in path

    # ── Subdomain analysis ────────────────────────────────────────────────────
    subdomain_count: int = 0     # Number of sub-levels (www.mail.evil = 2)
    has_www: int = 0             # 1 = exactly "www" subdomain

    # ── Domain entropy & digit ratio ─────────────────────────────────────────
    domain_entropy: float = 0.0  # Shannon entropy of registered domain token
    digit_ratio: float = 0.0     # Fraction of digits in domain token

    # ── Keyword / brand signals ───────────────────────────────────────────────
    keyword_count: int = 0       # Suspicious keywords found in full URL
    brand_in_url: int = 0        # Brand name appears in URL but not in domain
    path_depth: int = 0          # Number of non-empty path segments

    # ── Encoding & TLD ───────────────────────────────────────────────────────
    has_encoded_chars: int = 0   # 1 = % present (URL encoding)
    suspicious_tld: int = 0      # 1 = .tk / .xyz / etc.

    def to_vector(self) -> list[float]:
        """Return ordered list of feature values for the ML model."""
        return [float(v) for v in asdict(self).values()]

    @staticmethod
    def feature_names() -> list[str]:
        return list(Stage1Features.__dataclass_fields__.keys())


@dataclass
class Stage2Features:
    """
    Deep behavioral features extracted in Stage 2 when Stage 1 returns a
    'grey-zone' score (65 ≤ S1 < 90).  These simulate what would be gathered
    from a real HTTP request + HTML parse in a production system.
    """
    # ── Form analysis ─────────────────────────────────────────────────────────
    has_login_form: int = 0           # Login / password form detected
    external_form_action: int = 0     # Form submits to different domain
    has_redirect_param: int = 0       # ?redirect= or ?return= in query string

    # ── Obfuscation ───────────────────────────────────────────────────────────
    encoded_domain: int = 0           # xn-- punycode / IDN domain
    multiple_http: int = 0            # http:// appears > 1 time in URL
    base64_in_query: int = 0          # Long base64 token in query string
    double_encoded: int = 0           # %25 (double percent-encoding)

    # ── Deception tricks ─────────────────────────────────────────────────────
    fake_https_prefix: int = 0        # "https-" or "https." NOT at scheme position
    unicode_in_domain: int = 0        # Non-ASCII chars in domain (homograph)
    long_subdomain: int = 0           # Subdomain > 20 chars
    suspicious_path_keywords: int = 0 # verify/secure/update in path segments

    # ── Aggregated counts from S1 + S2 ────────────────────────────────────────
    # These are copies of the most informative S1 fields, re-used as input
    # to the Stage 2 model so it has the full picture.
    s1_keyword_count: int = 0
    s1_brand_in_url: int = 0
    s1_has_ip: int = 0
    s1_domain_entropy: float = 0.0
    s1_hyphen_count: int = 0

    def to_vector(self) -> list[float]:
        return [float(v) for v in asdict(self).values()]

    @staticmethod
    def feature_names() -> list[str]:
        return list(Stage2Features.__dataclass_fields__.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 3. STAGE 1 FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_stage1(url: str) -> Stage1Features:
    """
    Parse `url` and fill a Stage1Features dataclass.

    This function is designed to be:
      • Pure (no I/O, no side effects)
      • Fast (< 1 ms per URL on modern hardware)
      • Deterministic
    """
    f = Stage1Features()
    parsed = urlparse(url)
    ext    = tldextract.extract(url)

    domain_token  = ext.domain or ""          # e.g. "paypal" in "paypal.com"
    registered    = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    subdomain     = ext.subdomain or ""
    path          = parsed.path or ""
    query         = parsed.query or ""
    url_lower     = url.lower()
    netloc        = parsed.netloc or ""

    # ── Lengths ───────────────────────────────────────────────────────────────
    f.url_length       = len(url)
    f.domain_length    = len(registered)
    f.subdomain_length = len(subdomain)
    f.path_length      = len(path)
    f.query_length     = len(query)

    # ── Character counts ─────────────────────────────────────────────────────
    f.dot_count        = url.count(".")
    f.slash_count      = url.count("/")
    f.hyphen_count     = url.count("-")
    f.at_count         = url.count("@")
    f.question_count   = url.count("?")
    f.equals_count     = url.count("=")
    f.underscore_count = url.count("_")
    f.tilde_count      = url.count("~")
    f.percent_count    = url.count("%")
    f.amp_count        = url.count("&")
    f.hash_count       = url.count("#")

    # ── Structural flags ─────────────────────────────────────────────────────
    f.has_https         = 1 if parsed.scheme == "https" else 0
    f.has_port          = 1 if (parsed.port and parsed.port not in (80, 443)) else 0
    f.has_ip            = 1 if _IP_RE.match(parsed.hostname or "") else 0
    f.is_shortener      = 1 if registered in URL_SHORTENERS else 0
    f.double_slash_path = 1 if "//" in path else 0

    path_lower = path.lower()
    f.has_suspicious_ext = 1 if any(path_lower.endswith(e) for e in SUSPICIOUS_FILE_EXTS) else 0

    # ── Subdomain analysis ────────────────────────────────────────────────────
    sub_parts        = [p for p in subdomain.split(".") if p]
    f.subdomain_count = len(sub_parts)
    f.has_www         = 1 if subdomain == "www" else 0

    # ── Entropy of the domain token ───────────────────────────────────────────
    if domain_token:
        freq = {c: domain_token.count(c) / len(domain_token) for c in set(domain_token)}
        f.domain_entropy = round(
            -sum(p * math.log2(p) for p in freq.values() if p > 0), 4
        )
        f.digit_ratio = round(
            sum(c.isdigit() for c in domain_token) / len(domain_token), 4
        )

    # ── Keyword / brand signals ───────────────────────────────────────────────
    f.keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

    registered_lower = registered.lower()
    f.brand_in_url = sum(
        1 for brand in BRAND_NAMES
        if brand in url_lower and brand not in registered_lower
    )

    # ── Path depth ────────────────────────────────────────────────────────────
    f.path_depth = len([seg for seg in path.split("/") if seg])

    # ── Encoding & TLD ───────────────────────────────────────────────────────
    f.has_encoded_chars = 1 if "%" in url else 0
    f.suspicious_tld    = 1 if any(
        f".{ext.suffix}".endswith(t) for t in SUSPICIOUS_TLDS
    ) else 0

    return f


# ─────────────────────────────────────────────────────────────────────────────
# 4. STAGE 2 FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_stage2(url: str, s1: Stage1Features) -> Stage2Features:
    """
    Extract deeper, behavioral features.

    In production this function would:
      1. Fetch the URL with a headless browser / requests session
      2. Parse the HTML for <form> tags, iframes, JS redirects
      3. Follow redirects and track the chain length

    For now we derive all features purely from the URL string and the
    already-computed S1 features.  This is clearly documented so a
    contributor can swap in real HTTP analysis.
    """
    f   = Stage2Features()
    parsed    = urlparse(url)
    ext       = tldextract.extract(url)
    url_lower = url.lower()
    query_l   = (parsed.query or "").lower()
    path_l    = (parsed.path or "").lower()

    # ── Form / credential harvesting signals ─────────────────────────────────
    f.has_login_form = 1 if any(
        kw in url_lower for kw in ("login", "signin", "password", "pwd", "credential")
    ) else 0

    f.external_form_action = 1 if any(
        kw in query_l for kw in ("redirect", "return", "next", "goto", "url=")
    ) else 0

    f.has_redirect_param = f.external_form_action  # same signal

    # ── Obfuscation ───────────────────────────────────────────────────────────
    f.encoded_domain    = 1 if "xn--" in url_lower else 0
    f.multiple_http     = 1 if url_lower.count("http") > 1 else 0
    f.base64_in_query   = 1 if _BASE64_RE.search(parsed.query or "") else 0
    f.double_encoded    = 1 if "%25" in url else 0

    # ── Deception tricks ─────────────────────────────────────────────────────
    # Fake-HTTPS: "https-" or "https." appearing AFTER the scheme position
    stripped = url_lower.replace("https://", "", 1).replace("http://", "", 1)
    f.fake_https_prefix = 1 if ("https-" in stripped or "https." in stripped) else 0

    f.unicode_in_domain = 1 if any(ord(c) > 127 for c in ext.domain or "") else 0

    subdomain = ext.subdomain or ""
    f.long_subdomain = 1 if len(subdomain) > 20 else 0

    f.suspicious_path_keywords = 1 if any(
        kw in path_l for kw in ("secure", "verify", "update", "confirm", "account")
    ) else 0

    # ── Carry over the most important S1 signals ─────────────────────────────
    f.s1_keyword_count  = s1.keyword_count
    f.s1_brand_in_url   = s1.brand_in_url
    f.s1_has_ip         = s1.has_ip
    f.s1_domain_entropy = s1.domain_entropy
    f.s1_hyphen_count   = s1.hyphen_count

    return f
