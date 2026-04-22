"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/ml/dataset.py                       ║
║         Real Dataset Builder — Multi-Source + Tranco         ║
╚══════════════════════════════════════════════════════════════╝
 
Supported phishing data sources (tried in priority order):
  1. Kaggle       — phishing_site_urls.csv   (best, 11k+ URLs)
  2. PhishStats   — phishstats.csv           (no signup needed)
  3. OpenPhish    — openphish.txt            (live feed, no signup)
  4. PhishTank    — phishtank.csv            (if available)
  5. Built-in fallback                       (always available)
 
Quick start (no signup needed):
  # Option A — PhishStats
  Invoke-WebRequest -Uri "https://phishstats.info/phish_score.csv" -OutFile "backend\\data\\phishstats.csv"
 
  # Option B — OpenPhish
  Invoke-WebRequest -Uri "https://openphish.com/feed.txt" -OutFile "backend\\data\\openphish.txt"
 
  # Option C — Kaggle (best quality, free account)
  # Download from: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
  # Place at: backend/data/phishing_site_urls.csv
 
Usage
─────
  python -m app.ml.dataset                          # auto-detect source
  python -m app.ml.dataset --max-phishing 10000     # larger dataset
  python -m app.ml.dataset --source kaggle          # force specific source
  python -m app.ml.dataset --source phishstats
  python -m app.ml.dataset --source openphish
"""
 
import csv
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
 
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
 
from app.ml.features import (
    Stage1Features,
    Stage2Features,
    extract_stage1,
    extract_stage2,
)
 
logger = logging.getLogger("phishguard.dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
 
# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
 
BASE_DIR       = Path(__file__).resolve().parents[2]
DATA_DIR       = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
 
KAGGLE_CSV     = DATA_DIR / "phishing_site_urls.csv"
PHISHSTATS_CSV = DATA_DIR / "phishstats.csv"
OPENPHISH_TXT  = DATA_DIR / "openphish.txt"
PHISHTANK_CSV  = DATA_DIR / "phishtank.csv"
TRANCO_CSV     = DATA_DIR / "top-1m.csv"
OUTPUT_CSV     = DATA_DIR / "phishguard_dataset.csv"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. PHISHING URL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
 
def load_kaggle(path: Path, max_rows: int = 10_000) -> list:
    """
    Kaggle 'Phishing Site URLs' dataset.
    Download: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
    Columns: URL, Label  ('bad' = phishing, 'good' = safe)
    """
    if not path.exists():
        return []
    urls = []
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        url_col   = next((c for c in df.columns if "url"   in c), None)
        label_col = next((c for c in df.columns if "label" in c or "type" in c), None)
        if url_col is None:
            logger.warning("[Kaggle] No URL column found in file")
            return []
        for _, row in df.iterrows():
            url   = str(row.get(url_col,   "") or "").strip()
            label = str(row.get(label_col, "bad") or "bad").strip().lower() if label_col else "bad"
            if label_col and label not in ("bad", "phishing", "1", "malicious"):
                continue
            if url and not url.startswith("http"):
                url = "http://" + url
            if url.startswith(("http://", "https://")):
                urls.append(url)
            if len(urls) >= max_rows:
                break
        logger.info(f"[Kaggle]     Loaded {len(urls):,} phishing URLs from {path.name}")
    except Exception as e:
        logger.warning(f"[Kaggle] Error: {e}")
    return urls
 
 
def load_openphish(path: Path, max_rows: int = 10_000) -> list:
    """
    OpenPhish plain-text feed — one URL per line.
    Download: https://openphish.com/feed.txt  (no signup, updated every 6h)
 
    PowerShell:
      Invoke-WebRequest -Uri "https://openphish.com/feed.txt" -OutFile "backend\\data\\openphish.txt"
    """
    if not path.exists():
        return []
    urls = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                url = line.strip()
                if url.startswith(("http://", "https://")):
                    urls.append(url)
                if len(urls) >= max_rows:
                    break
        logger.info(f"[OpenPhish]  Loaded {len(urls):,} phishing URLs from {path.name}")
    except Exception as e:
        logger.warning(f"[OpenPhish] Error: {e}")
    return urls
 
 
def load_phishstats(path: Path, max_rows: int = 10_000) -> list:
    """
    PhishStats CSV — columns: #date, score, url, ip
    Download: https://phishstats.info/phish_score.csv  (no signup)
 
    PowerShell:
      Invoke-WebRequest -Uri "https://phishstats.info/phish_score.csv" -OutFile "backend\\data\\phishstats.csv"
    """
    if not path.exists():
        return []
    urls = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                # Skip comment/header lines
                if row[0].strip().startswith("#") or "score" in row[1].lower():
                    continue
                url = row[2].strip()
                if url.startswith(("http://", "https://")):
                    urls.append(url)
                if len(urls) >= max_rows:
                    break
        logger.info(f"[PhishStats] Loaded {len(urls):,} phishing URLs from {path.name}")
    except Exception as e:
        logger.warning(f"[PhishStats] Error: {e}")
    return urls
 
 
def load_phishtank(path: Path, max_rows: int = 10_000) -> list:
    """PhishTank CSV (legacy, kept for backward compatibility)."""
    if not path.exists():
        return []
    urls = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = (row.get("url") or "").strip()
                if url.startswith(("http://", "https://")):
                    urls.append(url)
                if len(urls) >= max_rows:
                    break
        logger.info(f"[PhishTank]  Loaded {len(urls):,} phishing URLs from {path.name}")
    except Exception as e:
        logger.warning(f"[PhishTank] Error: {e}")
    return urls
 
 
def load_phishing_urls(max_rows: int = 10_000, source: str = "auto") -> tuple:
    """
    Master loader — tries all available phishing sources in priority order.
    Returns (urls, source_name).
    """
    all_sources = [
        ("kaggle",     lambda: load_kaggle(KAGGLE_CSV, max_rows)),
        ("phishstats", lambda: load_phishstats(PHISHSTATS_CSV, max_rows)),
        ("openphish",  lambda: load_openphish(OPENPHISH_TXT, max_rows)),
        ("phishtank",  lambda: load_phishtank(PHISHTANK_CSV, max_rows)),
    ]
 
    if source != "auto":
        all_sources = [(n, fn) for n, fn in all_sources if n == source]
 
    for name, loader in all_sources:
        urls = loader()
        if urls:
            logger.info(f"[SOURCE] Using '{name}' as phishing data source")
            return urls, name
 
    logger.warning("=" * 55)
    logger.warning("  No real phishing data found — using built-in fallback")
    logger.warning("  For better accuracy, place ONE of these in backend/data/:")
    logger.warning("  • phishing_site_urls.csv  ← Kaggle (recommended)")
    logger.warning("    https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls")
    logger.warning("  • phishstats.csv          ← no signup needed")
    logger.warning("    Invoke-WebRequest -Uri 'https://phishstats.info/phish_score.csv' -OutFile 'backend\\data\\phishstats.csv'")
    logger.warning("  • openphish.txt           ← no signup needed")
    logger.warning("    Invoke-WebRequest -Uri 'https://openphish.com/feed.txt' -OutFile 'backend\\data\\openphish.txt'")
    logger.warning("=" * 55)
    return [], "fallback"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. SAFE URL LOADER
# ─────────────────────────────────────────────────────────────────────────────
 
def load_tranco(path: Path, max_rows: int = 10_000) -> list:
    """
    Load safe domains from Tranco top-1M list.
    Download: https://tranco-list.eu/top-1m.csv.zip  (no signup)
 
    PowerShell:
      Invoke-WebRequest -Uri "https://tranco-list.eu/top-1m.csv.zip" -OutFile "tranco.zip"
      Expand-Archive "tranco.zip" -DestinationPath "backend\\data\\"
    """
    if not path.exists():
        return []
    urls = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    domain = row[1].strip().lower()
                    if domain and "." in domain:
                        urls.append(f"https://{domain}")
                if len(urls) >= max_rows:
                    break
        logger.info(f"[Tranco]     Loaded {len(urls):,} safe domains from {path.name}")
    except Exception as e:
        logger.warning(f"[Tranco] Error: {e}")
    return urls
 
 
def load_safe_urls_from_kaggle(path: Path, max_rows: int = 10_000) -> list:
    """
    Extract SAFE URLs from the Kaggle dataset (Label == 'good').
    This gives real safe URLs instead of just bare domains.
    """
    if not path.exists():
        return []
    urls = []
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        url_col   = next((c for c in df.columns if "url"   in c), None)
        label_col = next((c for c in df.columns if "label" in c or "type" in c), None)
        if url_col is None or label_col is None:
            return []
        for _, row in df.iterrows():
            url   = str(row.get(url_col,   "") or "").strip()
            label = str(row.get(label_col, "bad") or "bad").strip().lower()
            if label not in ("good", "safe", "0", "legitimate", "benign"):
                continue
            if url and not url.startswith("http"):
                url = "http://" + url
            if url.startswith(("http://", "https://")):
                urls.append(url)
            if len(urls) >= max_rows:
                break
        logger.info(f"[Kaggle-Safe] Loaded {len(urls):,} safe URLs from {path.name}")
    except Exception as e:
        logger.warning(f"[Kaggle-Safe] Error: {e}")
    return urls
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. FALLBACK DATA (when no real source is available)
# ─────────────────────────────────────────────────────────────────────────────
 
def load_fallback_safe_urls(n: int = 500) -> list:
    """Hard realistic safe URLs — includes login pages to challenge the model."""
    hard_safe = [
        "https://accounts.google.com/signin/v2/identifier?flowName=GlifWebSignIn",
        "https://www.facebook.com/login/?next=https%3A%2F%2Fwww.facebook.com%2F",
        "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "https://appleid.apple.com/auth/authorize?response_type=code",
        "https://www.amazon.com/ap/signin?openid.return_to=https%3A%2F%2Fwww.amazon.com",
        "https://secure.paypal.com/us/signin?intent=checkout",
        "https://www.linkedin.com/uas/login?session_redirect=&fromSignIn=true",
        "https://github.com/login?return_to=%2Fuser%2Fsettings%2Fsecurity",
        "https://app.slack.com/signin/verify",
        "https://zoom.us/signin#/login",
        "https://www.amazon.com/dp/B09X7CRKRZ?ref=cm_sw_r_tw_ud_dp&tag=affiliate",
        "https://www.google.com/search?q=phishing+detection&rlz=1C1GCEA&oq=phishing",
        "https://stackoverflow.com/questions/tagged/machine-learning?tab=newest&page=2",
        "https://www.chase.com/personal/banking/online-banking/login",
        "https://www.wellsfargo.com/online-banking/login/?LOB=CONS",
        "https://console.aws.amazon.com/iam/home?region=us-east-1",
        "https://portal.azure.com/#blade/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade",
        "https://mail.google.com/mail/u/0/#inbox",
        "https://myaccount.google.com/security-checkup/all?continue=",
        "https://support.apple.com/en-us/HT204074",
        "https://help.netflix.com/en/node/412",
        "https://developer.github.com/v3/oauth/#web-application-flow",
    ]
    base_domains = [
        "google.com", "youtube.com", "facebook.com", "twitter.com",
        "instagram.com", "linkedin.com", "github.com", "microsoft.com",
        "apple.com", "amazon.com", "netflix.com", "wikipedia.org",
        "reddit.com", "stackoverflow.com", "cloudflare.com", "bbc.co.uk",
        "cnn.com", "nytimes.com", "reuters.com", "python.org",
        "npmjs.com", "pypi.org", "docker.com", "mozilla.org",
        "dropbox.com", "outlook.com", "discord.com", "spotify.com",
        "stripe.com", "shopify.com", "ebay.com", "adobe.com",
    ]
    paths = [
        "/", "/about", "/contact", "/help", "/support",
        "/blog/2024/security-best-practices",
        "/docs/api/v2/reference",
        "/account/settings?tab=security",
        "/search?q=how+to+stay+safe+online&page=2",
    ]
    augmented = list(hard_safe)
    for domain in base_domains:
        for path in paths:
            augmented.append(f"https://{domain}{path}")
    random.shuffle(augmented)
    return augmented[:n]
 
 
def load_fallback_phishing_urls(n: int = 500) -> list:
    """Realistic phishing URL patterns covering all major attack types."""
    def rand_str(length=8):
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=length))
 
    def rand_ip():
        return f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
 
    templates = [
        lambda: f"http://paypal-secure-{rand_str(5)}.tk/verify/account?user={rand_str(8)}@gmail.com",
        lambda: f"https://amazon-prize-winner-{rand_str(4)}.xyz/claim?ref={rand_str(12)}",
        lambda: f"http://apple-id-confirm-{rand_str(5)}.info/signin?token={rand_str(16)}",
        lambda: f"https://microsoft-support-{rand_str(4)}.net/alert/virus-detected.php",
        lambda: f"http://bankofamerica-secure-{rand_str(5)}.com/account/verify",
        lambda: f"https://facebook-login-{rand_str(4)}.ml/checkpoint/verify?next=/home",
        lambda: f"http://netflix-billing-update-{rand_str(4)}.cc/payment.php?session={rand_str(12)}",
        lambda: f"https://secure-login-paypal-{rand_str(4)}.top/webscr?cmd=_login-submit",
        lambda: f"https://login-{rand_str(6)}.secure-verify.xyz/account/signin",
        lambda: f"https://account-verify-{rand_str(5)}.info/google/oauth2/signin",
        lambda: f"http://{rand_ip()}/login/paypal/verify.php?email={rand_str(8)}@yahoo.com",
        lambda: f"http://{rand_ip()}:8080/google/account/signin?redirect={rand_str(8)}",
        lambda: f"http://paypal.com.{rand_str(8)}.ru/signin",
        lambda: f"https://accounts.google.com.{rand_str(6)}.tk/o/oauth2/auth",
        lambda: f"http://{rand_str(12)}.{random.choice(['tk','ml','ga','cf'])}/login?ref={rand_str(8)}",
        lambda: f"https://{rand_str(10)}-{rand_str(6)}.xyz/secure/verify?id={rand_str(16)}",
        lambda: f"https://xn--pypal-4ve.com/signin/verify",
        lambda: f"http://www.payp%61l.com/verify?token={rand_str(12)}",
        lambda: f"http://{rand_str(8)}.tk/go?url=http://paypal-verify.{rand_str(4)}.ml/login",
        lambda: f"https://www.{rand_str(6)}-bank.com/online-banking/login/verify",
    ]
    urls = []
    for i in range(n):
        fn = templates[i % len(templates)]
        try:
            urls.append(fn())
        except Exception:
            urls.append(f"http://phishing-{rand_str(8)}.tk/login")
    return urls
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_features_batch(urls: list, label: int, batch_size: int = 100, add_noise: bool = True) -> list:
    """Extract Stage1+Stage2 features for all URLs with optional noise injection."""
    records = []
    total   = len(urls)
    errors  = 0
 
    s1_names = Stage1Features.feature_names()
    s2_names = Stage2Features.feature_names()
 
    binary_features = {
        "has_https", "has_port", "has_ip", "is_shortener", "double_slash_path",
        "has_suspicious_ext", "has_www", "has_encoded_chars", "suspicious_tld",
        "has_login_form", "external_form_action", "has_redirect_param",
        "encoded_domain", "multiple_http", "base64_in_query", "double_encoded",
        "fake_https_prefix", "unicode_in_domain", "long_subdomain",
        "suspicious_path_keywords",
    }
 
    for i, url in enumerate(urls):
        if i % batch_size == 0:
            pct = (i / total) * 100
            logger.info(f"  Extracting features: {i:,}/{total:,}  ({pct:.1f}%)  errors={errors}")
        try:
            s1 = extract_stage1(url)
            s2 = extract_stage2(url, s1)
            record = {"url": url, "label": label}
            for name, val in zip(s1_names, s1.to_vector()):
                if add_noise and name not in binary_features:
                    val = max(0.0, float(val) + np.random.normal(0, abs(float(val)) * 0.05 + 0.1))
                record[f"s1_{name}"] = val
            for name, val in zip(s2_names, s2.to_vector()):
                if add_noise and name not in binary_features:
                    val = max(0.0, float(val) + np.random.normal(0, abs(float(val)) * 0.05 + 0.1))
                record[f"s2_{name}"] = val
            if add_noise and random.random() < 0.02:
                record["label"] = 1 - label   # 2% label noise
            records.append(record)
        except Exception as e:
            errors += 1
            logger.debug(f"  Skipped {url}: {e}")
 
    logger.info(f"  Done. {len(records):,} records extracted, {errors} skipped.")
    return records
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
 
def build_dataset(
    max_phishing: int = 10_000,
    max_safe: int     = 10_000,
    output_path: Path = OUTPUT_CSV,
    source: str       = "auto",
) -> "pd.DataFrame":
    import pandas as pd
 
    logger.info("=" * 60)
    logger.info("  PhishGuard Dataset Builder")
    logger.info("=" * 60)
 
    # ── Phishing URLs ─────────────────────────────────────────────────────────
    phishing_urls, used_source = load_phishing_urls(max_rows=max_phishing, source=source)
    if not phishing_urls:
        phishing_urls = load_fallback_phishing_urls(min(max_phishing, 1000))
        used_source   = "fallback"
 
    # ── Safe URLs ─────────────────────────────────────────────────────────────
    # If Kaggle data is available, extract safe URLs from it too (more realistic)
    safe_urls = load_safe_urls_from_kaggle(KAGGLE_CSV, max_rows=max_safe) if used_source == "kaggle" else []
    if not safe_urls:
        safe_urls = load_tranco(TRANCO_CSV, max_rows=max_safe)
    if not safe_urls:
        logger.warning("Tranco not found — using built-in safe URL fallback")
        safe_urls = load_fallback_safe_urls(min(max_safe, 1000))
 
    # ── Balance classes ───────────────────────────────────────────────────────
    n = min(len(phishing_urls), len(safe_urls))
    logger.info(f"Balancing to {n:,} samples per class  ({n*2:,} total)")
    random.shuffle(phishing_urls)
    random.shuffle(safe_urls)
    phishing_urls = phishing_urls[:n]
    safe_urls     = safe_urls[:n]
 
    # ── Feature extraction ────────────────────────────────────────────────────
    logger.info(f"Extracting features for {n:,} phishing URLs…")
    t0 = time.perf_counter()
    phishing_records = extract_features_batch(phishing_urls, label=1)
 
    logger.info(f"Extracting features for {n:,} safe URLs…")
    safe_records = extract_features_batch(safe_urls, label=0)
 
    logger.info(f"Feature extraction complete in {time.perf_counter()-t0:.1f}s")
 
    # ── Assemble + save ───────────────────────────────────────────────────────
    all_records = phishing_records + safe_records
    random.shuffle(all_records)
    df = pd.DataFrame(all_records)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved → {output_path}  ({len(df):,} rows, {len(df.columns)} columns)")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 50)
    logger.info("  DATASET SUMMARY")
    logger.info("─" * 50)
    logger.info(f"  Source        : {used_source}")
    logger.info(f"  Total rows    : {len(df):,}")
    logger.info(f"  Safe (0)      : {(df['label']==0).sum():,}")
    logger.info(f"  Phishing (1)  : {(df['label']==1).sum():,}")
    logger.info(f"  Total features: {len(df.columns) - 2}")
 
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numeric_cols = [c for c in df.columns if c not in ("url", "label")]
        correlations = df[numeric_cols].corrwith(df["label"]).abs().sort_values(ascending=False)
        logger.info("")
        logger.info("  Top 10 features by correlation with phishing label:")
        for feat, corr in correlations.head(10).items():
            bar = "█" * int(corr * 20)
            logger.info(f"    {feat:<35} {corr:.3f}  {bar}")
    logger.info("─" * 50)
 
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. LOADER (used by model.py at training time)
# ─────────────────────────────────────────────────────────────────────────────
 
def load_dataset(path: Path = OUTPUT_CSV):
    """Load pre-built dataset CSV. Returns (X_s1, X_s2, y) or None."""
    import pandas as pd
    if not path.exists():
        return None
    logger.info(f"[DATASET] Loading real dataset from {path.name}…")
    df = pd.read_csv(path)
    s1_cols = [c for c in df.columns if c.startswith("s1_")]
    s2_cols = [c for c in df.columns if c.startswith("s2_")]
    if not s1_cols:
        logger.warning("[DATASET] No feature columns found — check CSV format")
        return None
    X_s1 = df[s1_cols].fillna(0).values.astype(float)
    X_s2 = df[s1_cols + s2_cols].fillna(0).values.astype(float)
    y    = df["label"].values.astype(int)
    logger.info(f"[DATASET] {len(y):,} samples | S1={X_s1.shape[1]} feats | S2={X_s2.shape[1]} feats")
    return X_s1, X_s2, y
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build PhishGuard training dataset")
    parser.add_argument("--max-phishing", type=int,  default=10_000)
    parser.add_argument("--max-safe",     type=int,  default=10_000)
    parser.add_argument("--output",       type=str,  default=str(OUTPUT_CSV))
    parser.add_argument("--source",       type=str,  default="auto",
                        choices=["auto","kaggle","phishstats","openphish","phishtank"])
    args = parser.parse_args()
    build_dataset(
        max_phishing=args.max_phishing,
        max_safe=args.max_safe,
        output_path=Path(args.output),
        source=args.source,
    )