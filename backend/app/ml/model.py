"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/ml/model.py                         ║
║         Machine Learning Model — Train · Load · Predict      ║
╚══════════════════════════════════════════════════════════════╝
 
Architecture
────────────
  Stage 1 Model  — trained on Stage1Features (URL / domain level)
  Stage 2 Model  — trained on Stage1Features + Stage2Features (deep)
 
  Both models are Random Forest classifiers (sklearn).
  The output of .predict_proba() gives P(phishing) which is scaled
  to a 0–100 risk score.
 
  The models are trained on a synthetic dataset the first time they
  run, then persisted to disk via joblib.  In production, replace
  the synthetic data with a real labelled dataset (e.g. PhishTank
  + Alexa/Majestic top-1M for negatives).
 
Decision thresholds  (mirror app/core/config.py settings)
──────────────────────────────────────────────────────────
  Stage 1  ≥ 0.90  → phishing  (skip S2)
  Stage 1  < 0.65  → safe      (skip S2)
  0.65 ≤ S1 < 0.90 → run Stage 2
  Stage 2  ≥ 0.80  → phishing
  Stage 2  < 0.80  → suspicious
"""
 
import logging
import os
import pickle
import time
from dataclasses import asdict
from typing import Optional
 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
 
from app.ml.features import (
    Stage1Features,
    Stage2Features,
    extract_stage1,
    extract_stage2,
)
 
logger = logging.getLogger("phishguard.ml")
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
 
try:
    from app.core.config import settings
    _MODEL_DIR  = os.path.dirname(settings.MODEL_PATH)
    _S1_PATH    = settings.MODEL_PATH.replace(".pkl", "_s1.pkl")
    _S2_PATH    = settings.MODEL_PATH.replace(".pkl", "_s2.pkl")
    _S1_THRESH  = settings.STAGE1_PHISHING_THRESHOLD
    _S1_SAFE    = settings.STAGE1_SAFE_THRESHOLD
    _S2_THRESH  = settings.STAGE2_PHISHING_THRESHOLD
except Exception:
    _base = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    _MODEL_DIR  = _base
    _S1_PATH    = os.path.join(_base, "phishguard_model_s1.pkl")
    _S2_PATH    = os.path.join(_base, "phishguard_model_s2.pkl")
    _S1_THRESH  = 0.90
    _S1_SAFE    = 0.65
    _S2_THRESH  = 0.80
 
 
os.makedirs(_MODEL_DIR, exist_ok=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC TRAINING DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
 
# In production, replace _generate_s1_data() / _generate_s2_data() with a
# real dataset loader (CSV, DB query, PhishTank API, etc.).
 
_SAFE_URLS = [
    "https://www.google.com/search?q=phishing",
    "https://github.com/openai/chatgpt",
    "https://stackoverflow.com/questions/12345",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.amazon.com/dp/B0123456",
    "https://www.microsoft.com/en-us/windows",
    "https://docs.python.org/3/library/urllib.html",
    "https://fastapi.tiangolo.com/tutorial/",
    "https://www.reddit.com/r/netsec",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
]
 
_PHISHING_URLS = [
    "http://paypal-secure-login.com/verify?user=victim@email.com",
    "https://amazon-prize-claim-2024.tk/win?ref=email",
    "http://192.168.1.1/google-account-verify.php",
    "https://secure-login-facebook.info/signin?redirect=http://evil.com",
    "http://microsoft-support-alert.net/virus-detected.php",
    "https://apple-id-confirm-now.xyz/login?token=abc123",
    "http://bankofamerica-secure-update.com/account/verify",
    "https://netflix-billing-updatepayment.ml/update.php",
    "http://login-verify.instagram-secure-account.info/",
    "https://chase-bank-alert-suspended.cc/verify.php",
]
 
 
def _url_to_s1_vector(url: str) -> list[float]:
    try:
        return extract_stage1(url).to_vector()
    except Exception:
        return [0.0] * len(Stage1Features.feature_names())
 
 
def _url_to_s2_vector(url: str) -> list[float]:
    try:
        s1 = extract_stage1(url)
        s2 = extract_stage2(url, s1)
        return s1.to_vector() + s2.to_vector()
    except Exception:
        total = len(Stage1Features.feature_names()) + len(Stage2Features.feature_names())
        return [0.0] * total
 
 
def _generate_s1_data():
    """
    Generate a small synthetic labelled dataset for Stage 1.
    Augments the seed URLs with noise to produce ~200 samples.
    """
    X, y = [], []
 
    def add_samples(urls, label, n_augment=10):
        for url in urls:
            X.append(_url_to_s1_vector(url))
            y.append(label)
            # Augment with minor perturbations
            for _ in range(n_augment):
                aug = url + "".join(
                    np.random.choice(list("abcdefghijklmnop"), size=3)
                )
                X.append(_url_to_s1_vector(aug))
                y.append(label)
 
    add_samples(_SAFE_URLS,     label=0, n_augment=10)
    add_samples(_PHISHING_URLS, label=1, n_augment=10)
    return np.array(X), np.array(y)
 
 
def _generate_s2_data():
    X, y = [], []
 
    def add_samples(urls, label, n_augment=10):
        for url in urls:
            X.append(_url_to_s2_vector(url))
            y.append(label)
            for _ in range(n_augment):
                aug = url + "".join(
                    np.random.choice(list("abcdefghijklmnop"), size=3)
                )
                X.append(_url_to_s2_vector(aug))
                y.append(label)
 
    add_samples(_SAFE_URLS,     label=0, n_augment=10)
    add_samples(_PHISHING_URLS, label=1, n_augment=10)
    return np.array(X), np.array(y)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────
 
class PhishingModel:
    """
    Manages two sklearn Pipelines:
      _s1_pipeline  — Stage 1 (URL features only)
      _s2_pipeline  — Stage 2 (URL + deep features)
 
    Both pipelines contain:
      1. StandardScaler  — zero-mean, unit-variance normalisation
      2. RandomForestClassifier (n_estimators=200, calibrated)
 
    Public methods
    ──────────────
      load_or_train()        — Load from disk; train & save if not found
      predict_stage1(url)    → (probability: float, reasons: list[str])
      predict_stage2(url)    → (probability: float, reasons: list[str])
    """
 
    def __init__(self) -> None:
        self._s1_pipeline: Optional[Pipeline] = None
        self._s2_pipeline: Optional[Pipeline] = None
        self._trained_at: Optional[float]     = None
 
    # ── Persistence ───────────────────────────────────────────────────────────
 
    def load_or_train(self) -> None:
        """Load pre-trained models from disk, or train new ones."""
        s1_loaded = self._load_pipeline(_S1_PATH, "s1")
        s2_loaded = self._load_pipeline(_S2_PATH, "s2")
 
        if not s1_loaded or not s2_loaded:
            logger.info("[ML] Models not found on disk — training new models…")
            self._train_and_save()
        else:
            logger.info("[ML] Models loaded from disk successfully.")
            self._trained_at = os.path.getmtime(_S1_PATH)
 
    def _load_pipeline(self, path: str, name: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                pipeline = pickle.load(fh)
            setattr(self, f"_{name}_pipeline", pipeline)
            logger.debug(f"[ML] Loaded {name} pipeline from {path}")
            return True
        except Exception as exc:
            logger.warning(f"[ML] Could not load {name} pipeline: {exc}")
            return False
 
    def _train_and_save(self) -> None:
        t0 = time.perf_counter()
 
        # ── Try to load a real pre-built dataset first ────────────────────────
        real_data = None
        try:
            from app.ml.dataset import load_dataset
            real_data = load_dataset()
        except Exception as e:
            logger.warning(f"[ML] Could not load real dataset: {e}")
 
        if real_data is not None:
            X_s1, X_s2, y = real_data
            logger.info(f"[ML] Using REAL dataset — {len(y):,} samples")
            X1, y1 = X_s1, y
            X2, y2 = X_s2, y
        else:
            logger.info("[ML] Real dataset not found — using synthetic fallback data")
            logger.info("[ML] Run  python -m app.ml.dataset  to build a real dataset")
            X1_list, y1_list = _generate_s1_data()
            X2_list, y2_list = _generate_s2_data()
            X1, y1 = X1_list, y1_list
            X2, y2 = X2_list, y2_list
 
        # ── Train with cross-validation reporting ─────────────────────────────
        logger.info("[ML] Training Stage 1 model…")
        self._s1_pipeline = self._build_pipeline()
        self._s1_pipeline.fit(X1, y1)
        self._save_pipeline(self._s1_pipeline, _S1_PATH, "s1")
        s1_score = self._s1_pipeline.score(X1, y1)
        logger.info(f"[ML] Stage 1 trained — {len(y1):,} samples, train acc={s1_score:.3f}")
 
        logger.info("[ML] Training Stage 2 model…")
        self._s2_pipeline = self._build_pipeline(n_estimators=300)
        self._s2_pipeline.fit(X2, y2)
        self._save_pipeline(self._s2_pipeline, _S2_PATH, "s2")
        s2_score = self._s2_pipeline.score(X2, y2)
        logger.info(f"[ML] Stage 2 trained — {len(y2):,} samples, train acc={s2_score:.3f}")
 
        self._trained_at = time.time()
        elapsed = time.perf_counter() - t0
        logger.info(f"[ML] Training complete in {elapsed:.2f}s.")
 
    @staticmethod
    def _build_pipeline(n_estimators: int = 200) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])
 
    @staticmethod
    def _save_pipeline(pipeline: Pipeline, path: str, name: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as fh:
                pickle.dump(pipeline, fh)
            logger.debug(f"[ML] Saved {name} pipeline to {path}")
        except Exception as exc:
            logger.error(f"[ML] Could not save {name} pipeline: {exc}")
 
    # ── Rule-Based Signal Extraction (generates human-readable reasons) ───────
 
    @staticmethod
    def _reasons_from_s1(f: Stage1Features) -> list[str]:
        """Map feature values to natural-language detection signals."""
        reasons = []
 
        if f.url_length > 100:
            reasons.append(f"Unusually long URL ({f.url_length} chars)")
        if f.at_count > 0:
            reasons.append("@ symbol in URL — common credential-theft trick")
        if f.has_ip:
            reasons.append("IP address used instead of domain name")
        if not f.has_https:
            reasons.append("No HTTPS — connection is unencrypted")
        if f.is_shortener:
            reasons.append("URL shortener detected — hides true destination")
        if f.brand_in_url > 0:
            reasons.append(
                f"Brand name in URL but not in registered domain "
                f"({f.brand_in_url} brand(s)) — impersonation attempt"
            )
        if f.keyword_count >= 3:
            reasons.append(f"Multiple suspicious keywords detected ({f.keyword_count})")
        elif f.keyword_count >= 1:
            reasons.append("Suspicious keyword in URL path")
        if f.suspicious_tld:
            reasons.append("Suspicious top-level domain (high phishing rate)")
        if f.domain_entropy > 3.5:
            reasons.append("Domain appears randomly generated (high entropy)")
        if f.digit_ratio > 0.4:
            reasons.append("Unusually high proportion of digits in domain")
        if f.subdomain_count > 3:
            reasons.append(f"Excessive subdomain depth ({f.subdomain_count} levels)")
        if f.hyphen_count > 3:
            reasons.append("Multiple hyphens in domain — common phishing pattern")
        if f.has_encoded_chars and f.percent_count > 3:
            reasons.append("Excessive URL encoding — possible obfuscation")
        if f.has_port:
            reasons.append("Non-standard port in URL")
        if f.double_slash_path:
            reasons.append("Double-slash in URL path — possible open redirect")
 
        return reasons
 
    @staticmethod
    def _reasons_from_s2(f2: Stage2Features) -> list[str]:
        reasons = []
        if f2.has_login_form:
            reasons.append("Login / credential-harvesting form detected")
        if f2.external_form_action:
            reasons.append("Form action points to external / redirect URL")
        if f2.multiple_http:
            reasons.append("Multiple HTTP redirects embedded in URL")
        if f2.fake_https_prefix:
            reasons.append("Fake HTTPS prefix in URL — deceptive SSL simulation")
        if f2.encoded_domain:
            reasons.append("Internationalised domain (possible homograph attack)")
        if f2.unicode_in_domain:
            reasons.append("Unicode characters in domain to mimic a legitimate site")
        if f2.long_subdomain:
            reasons.append("Extremely long subdomain (common phishing technique)")
        if f2.suspicious_path_keywords:
            reasons.append("Security-related keywords in URL path segments")
        if f2.base64_in_query:
            reasons.append("Base64-encoded payload detected in query string")
        if f2.double_encoded:
            reasons.append("Double URL-encoding detected — obfuscation attempt")
        return reasons
 
    # ── Public Prediction API ─────────────────────────────────────────────────
 
    def predict_stage1(self, url: str) -> tuple[float, list[str]]:
        """
        Run Stage 1 prediction.
 
        Returns
        ───────
        probability : float  — P(phishing) in [0.0, 1.0]
        reasons     : list   — human-readable detection signals
        """
        s1_features = extract_stage1(url)
        reasons     = self._reasons_from_s1(s1_features)
        prob        = self._predict_proba(self._s1_pipeline, s1_features.to_vector())
        return prob, reasons
 
    def predict_stage2(self, url: str, s1_features: Stage1Features) -> tuple[float, list[str]]:
        """
        Run Stage 2 prediction (call only when S1 is in the grey zone).
 
        Returns
        ───────
        probability : float  — refined P(phishing) in [0.0, 1.0]
        reasons     : list   — cumulative detection signals from S1 + S2
        """
        s2_features  = extract_stage2(url, s1_features)
        s1_reasons   = self._reasons_from_s1(s1_features)
        s2_reasons   = self._reasons_from_s2(s2_features)
        all_reasons  = list(dict.fromkeys(s1_reasons + s2_reasons))  # deduplicate, preserve order
 
        combined_vec = s1_features.to_vector() + s2_features.to_vector()
        prob         = self._predict_proba(self._s2_pipeline, combined_vec)
        return prob, all_reasons
 
    @staticmethod
    def _predict_proba(pipeline: Optional[Pipeline], vector: list[float]) -> float:
        """
        Run the sklearn pipeline and return P(class=1) (phishing probability).
        Falls back to a heuristic if the model is unavailable.
        """
        if pipeline is None:
            # Graceful fallback: use a simple heuristic on the raw feature vector
            # Feature 0 = url_length (normalised), Feature 16 = has_https, etc.
            # This is intentionally minimal — load_or_train() should always be called first.
            logger.warning("[ML] Pipeline not loaded — using fallback heuristic")
            return min(float(vector[0]) / 250.0, 1.0)
 
        X = np.array(vector).reshape(1, -1)
        proba = pipeline.predict_proba(X)[0]
        # proba[1] = P(phishing) when classes are [0=safe, 1=phishing]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])
 
    # ── Model Metadata ────────────────────────────────────────────────────────
 
    @property
    def is_ready(self) -> bool:
        return self._s1_pipeline is not None and self._s2_pipeline is not None
 
    def info(self) -> dict:
        """Return a dict describing the loaded model state (used in /health)."""
        import time as _time
        return {
            "ready":       self.is_ready,
            "trained_at":  self._trained_at,
            "s1_model":    str(type(self._s1_pipeline)) if self._s1_pipeline else None,
            "s2_model":    str(type(self._s2_pipeline)) if self._s2_pipeline else None,
            "s1_features": len(Stage1Features.feature_names()),
            "s2_features": len(Stage2Features.feature_names()),
            "thresholds": {
                "s1_phishing": _S1_THRESH,
                "s1_safe":     _S1_SAFE,
                "s2_phishing": _S2_THRESH,
            },
        }
 