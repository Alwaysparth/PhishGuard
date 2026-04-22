"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/ml/evaluate.py                      ║
║         Model Evaluation — Metrics, Confusion Matrix         ║
╚══════════════════════════════════════════════════════════════╝
 
Usage
─────
  # From inside backend/ directory:
  python -m app.ml.evaluate
 
  Outputs:
    • Accuracy, Precision, Recall, F1-Score
    • Confusion matrix
    • Per-class report
    • Feature importance rankings
    • Saves evaluation results to  data/evaluation_report.json
"""
 
import json
import logging
import sys
from pathlib import Path
 
import numpy as np
 
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
 
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
 
logger = logging.getLogger("phishguard.evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
 
DATA_DIR   = Path(__file__).resolve().parents[2] / "data"
REPORT_OUT = DATA_DIR / "evaluation_report.json"
 
 
def evaluate():
    # ── Load dataset ──────────────────────────────────────────────────────────
    from app.ml.dataset import load_dataset, build_dataset, OUTPUT_CSV
    from sklearn.model_selection import train_test_split
 
    data = load_dataset()
    if data is None:
        logger.info("No dataset found — building one now…")
        build_dataset(max_phishing=5000, max_safe=5000)
        data = load_dataset()
 
    if data is None:
        logger.error("Could not load or build dataset. Exiting.")
        return
 
    X_s1, X_s2, y = data
    logger.info(f"Dataset: {len(y):,} samples  |  "
                f"{(y==1).sum():,} phishing  |  {(y==0).sum():,} safe")
 
    # ── Stratified 80/20 train-test split ─────────────────────────────────────
    # Evaluation is done ONLY on the held-out test set — never on training data.
    # This is what prevents the 100% accuracy artifact.
    X_s1_train, X_s1_test, y_train, y_test = train_test_split(
        X_s1, y, test_size=0.20, random_state=42, stratify=y
    )
    X_s2_train, X_s2_test, _, _ = train_test_split(
        X_s2, y, test_size=0.20, random_state=42, stratify=y
    )
 
    logger.info(f"Train: {len(y_train):,}  |  Test (held-out): {len(y_test):,}")
 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_validate
 
    def make_pipeline(n=200):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n, max_depth=12,
                min_samples_split=4, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ])
 
    report = {}
 
    stages = [
        ("Stage 1 (URL features)",  X_s1_train, X_s1_test),
        ("Stage 2 (Deep features)", X_s2_train, X_s2_test),
    ]
 
    for stage_name, X_train, X_test in stages:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Evaluating: {stage_name}")
        logger.info(f"  Train size: {len(y_train):,}  |  Test size: {len(y_test):,}")
        logger.info(f"  Feature vector size: {X_train.shape[1]}")
        logger.info("=" * 60)
 
        pipeline = make_pipeline(200 if "1" in stage_name else 300)
 
        # ── 5-fold CV on TRAINING data only ───────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=True,   # show train vs val gap (overfitting signal)
            n_jobs=-1,
        )
 
        logger.info("")
        logger.info("  5-Fold CV on Training Data (val scores):")
        logger.info(f"  {'Metric':<15} {'Train':>8}  {'Val':>8}  {'Gap':>8}  Bar")
        logger.info(f"  {'-'*55}")
 
        cv_metrics = {}
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            train_mean = float(np.mean(cv_results[f"train_{m}"]))
            val_mean   = float(np.mean(cv_results[f"test_{m}"]))
            val_std    = float(np.std(cv_results[f"test_{m}"]))
            gap        = train_mean - val_mean
            bar        = "█" * int(val_mean * 25)
            # Flag suspicious overfitting
            flag = "  ⚠ OVERFIT" if gap > 0.05 else ""
            logger.info(f"  {m:<15} {train_mean:>8.4f}  {val_mean:>8.4f}  {gap:>+8.4f}  {bar}{flag}")
            cv_metrics[m] = {"train": round(train_mean, 4), "val": round(val_mean, 4),
                             "val_std": round(val_std, 4), "gap": round(gap, 4)}
 
        # ── Evaluate on HELD-OUT TEST SET ─────────────────────────────────────
        pipeline.fit(X_train, y_train)
        y_pred      = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
 
        test_acc  = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec  = recall_score(y_test, y_pred, zero_division=0)
        test_f1   = f1_score(y_test, y_pred, zero_division=0)
        test_auc  = roc_auc_score(y_test, y_pred_prob)
 
        logger.info("")
        logger.info("  ── Held-Out TEST SET Results (never seen during training) ──")
        logger.info(f"  Accuracy  : {test_acc:.4f}")
        logger.info(f"  Precision : {test_prec:.4f}  (of URLs flagged as phishing, how many really are)")
        logger.info(f"  Recall    : {test_rec:.4f}  (of actual phishing URLs, how many we caught)")
        logger.info(f"  F1-Score  : {test_f1:.4f}")
        logger.info(f"  ROC-AUC   : {test_auc:.4f}")
 
        # ── Confusion matrix ──────────────────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) * 100
        fnr = fn / (fn + tp) * 100
 
        logger.info("")
        logger.info("  Confusion Matrix (held-out test set):")
        logger.info(f"  {'':20} Predicted Safe   Predicted Phishing")
        logger.info(f"  {'Actual Safe':20} {tn:>14,}   {fp:>18,}   ← {fpr:.1f}% false alarm rate")
        logger.info(f"  {'Actual Phishing':20} {fn:>14,}   {tp:>18,}   ← {fnr:.1f}% miss rate")
 
        if test_acc == 1.0:
            logger.warning("")
            logger.warning("  ⚠  PERFECT SCORE ON TEST SET — this is suspicious.")
            logger.warning("  ⚠  Check: are your safe & phishing URL templates too different?")
            logger.warning("  ⚠  Use real PhishTank + Tranco data for realistic results.")
            logger.warning("  ⚠  Expected realistic ranges:  Acc=88-96%, F1=87-95%")
 
        # ── Feature importance ────────────────────────────────────────────────
        rf = pipeline.named_steps["clf"]
        importances = rf.feature_importances_
 
        from app.ml.features import Stage1Features, Stage2Features
        if "1" in stage_name:
            feat_names = [f"s1_{n}" for n in Stage1Features.feature_names()]
        else:
            feat_names = (
                [f"s1_{n}" for n in Stage1Features.feature_names()] +
                [f"s2_{n}" for n in Stage2Features.feature_names()]
            )
 
        top_idx = np.argsort(importances)[::-1][:10]
        logger.info("")
        logger.info("  Top 10 Feature Importances:")
        feat_importance_list = []
        for rank, idx in enumerate(top_idx, 1):
            name = feat_names[idx] if idx < len(feat_names) else f"feature_{idx}"
            imp  = importances[idx]
            bar  = "█" * int(imp * 100)
            logger.info(f"  {rank:>2}. {name:<35} {imp:.4f}  {bar}")
            feat_importance_list.append({"feature": name, "importance": round(float(imp), 5)})
 
        report[stage_name] = {
            "cv_metrics":           cv_metrics,
            "test_metrics": {
                "accuracy":  round(test_acc,  4),
                "precision": round(test_prec, 4),
                "recall":    round(test_rec,  4),
                "f1":        round(test_f1,   4),
                "roc_auc":   round(test_auc,  4),
            },
            "confusion_matrix":         {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "false_positive_rate_pct":  round(fpr, 2),
            "false_negative_rate_pct":  round(fnr, 2),
            "top_features":             feat_importance_list,
            "n_train":                  int(len(y_train)),
            "n_test":                   int(len(y_test)),
            "n_features":               int(X_train.shape[1]),
            "overfitting_warning":      test_acc == 1.0,
        }
 
    # ── Save report ───────────────────────────────────────────────────────────
    DATA_DIR.mkdir(exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("")
    logger.info(f"Evaluation report saved → {REPORT_OUT}")
 
    return report
 
 
if __name__ == "__main__":
    evaluate()