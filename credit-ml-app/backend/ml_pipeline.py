"""
ml_pipeline.py
--------------
Full ML training pipeline on the real German Credit Dataset.
Trains Random Forest, saves model artifacts, exposes predict().

Run standalone:
    python ml_pipeline.py
"""

import os, math, warnings, json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "german.data")
STORE_DIR   = os.path.join(BASE_DIR, "model_store")
RF_PATH     = os.path.join(STORE_DIR, "rf_model.pkl")
LR_PATH     = os.path.join(STORE_DIR, "lr_model.pkl")
SCALER_PATH = os.path.join(STORE_DIR, "scaler.pkl")
ENC_PATH    = os.path.join(STORE_DIR, "encoders.pkl")
META_PATH   = os.path.join(STORE_DIR, "model_meta.json")
os.makedirs(STORE_DIR, exist_ok=True)

# ── Column schema (exact German Credit Dataset columns) ────────
COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount",    "savings_account", "employment_since",
    "installment_rate", "personal_status", "other_debtors",
    "residence_since",  "property", "age", "other_plans",
    "housing",          "existing_credits", "job", "people_liable",
    "telephone",        "foreign_worker", "target",
]

CATEGORICAL_COLS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status", "other_debtors", "property",
    "other_plans",      "housing", "job", "telephone", "foreign_worker",
]

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age",      "existing_credits", "people_liable",
]

# After encoding, these are the extra repayment-timing features
TIMING_COLS = [
    "avg_days_early",    # positive = paid early, negative = paid late
    "repay_consistency", # 0.0–1.0, fraction of payments on time or early
    "early_pay_score",   # +10/+20/+30 reward points
    "delay_penalty",     # -5/-15/-35/-70 penalty points
    "net_timing_score",  # early_pay_score + delay_penalty
]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + TIMING_COLS

# ── 1. LOAD DATA ───────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", names=COLUMNS)
    # Recode: 1=Good→0 (will repay), 2=Bad→1 (will default)
    df["target"] = df["target"].map({1: 0, 2: 1})
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1] - 1} features")
    good = (df.target == 0).sum()
    bad  = (df.target == 1).sum()
    print(f"  Class split: {good} good ({good/len(df)*100:.0f}%) | {bad} bad ({bad/len(df)*100:.0f}%)")
    return df


# ── 2. ENCODE CATEGORICALS ─────────────────────────────────────

def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data     = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    return data, encoders


# ── 3. ADD REPAYMENT TIMING FEATURES ──────────────────────────
#   These are derived from user input at inference time.
#   During training we simulate realistic distributions
#   correlated with the target so the model learns the signal.

def add_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate repayment timing features correlated with creditworthiness.
    Good borrowers (target=0) tend to pay earlier and more consistently.
    Bad  borrowers (target=1) tend to pay later and less consistently.
    """
    np.random.seed(42)
    n = len(df)

    # avg_days_early: positive = paid before deadline
    #   Good: mean +7 days early | Bad: mean -8 days late
    avg_early = np.where(
        df["target"] == 0,
        np.random.normal(7,  8, n),   # good borrowers pay early
        np.random.normal(-8, 10, n),  # bad borrowers pay late
    )

    # repay_consistency: fraction of months paid on time/early
    consistency = np.where(
        df["target"] == 0,
        np.clip(np.random.normal(0.88, 0.10, n), 0.5, 1.0),
        np.clip(np.random.normal(0.55, 0.20, n), 0.0, 0.95),
    )

    # Derived reward / penalty scores (your innovation)
    early_pts = np.where(avg_early >= 10, 30,
                np.where(avg_early >= 5,  20,
                np.where(avg_early >= 1,  10, 0)))

    max_late  = np.abs(np.minimum(avg_early, 0))
    late_pts  = np.where(max_late >= 30, -70,
                np.where(max_late >= 8,  -35,
                np.where(max_late >= 1,  -15, 0)))

    df = df.copy()
    df["avg_days_early"]    = avg_early.round(2)
    df["repay_consistency"] = consistency.round(4)
    df["early_pay_score"]   = early_pts
    df["delay_penalty"]     = late_pts
    df["net_timing_score"]  = early_pts + late_pts
    return df


# ── 4. SPLIT & SCALE ───────────────────────────────────────────

def split_scale(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=FEATURE_COLS, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),
                               columns=FEATURE_COLS, index=X_test.index)

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ── 5. TRAIN ───────────────────────────────────────────────────

def train_rf(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators    = 300,
        max_depth       = 10,
        min_samples_leaf= 2,
        max_features    = "sqrt",
        class_weight    = "balanced",
        random_state    = 42,
        n_jobs          = -1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_lr(X_train, y_train) -> LogisticRegression:
    lr = LogisticRegression(C=0.5, max_iter=1000,
                             class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    return lr


# ── 6. EVALUATE ────────────────────────────────────────────────

def evaluate(model, X_test, y_test, name="Model", threshold=0.45) -> dict:
    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= threshold).astype(int)
    auc     = roc_auc_score(y_test, y_prob)
    acc     = accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n  {name}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Good","Bad"], zero_division=0))
    return dict(auc=auc, acc=acc, precision=prec,
                recall=rec, f1=f1, confusion_matrix=cm)


def cross_val(model, X, y, name="Model") -> dict:
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv,
                              scoring="roc_auc", n_jobs=-1)
    print(f"  {name} CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return {"mean": round(float(scores.mean()), 4),
            "std":  round(float(scores.std()),  4)}


# ── 7. CREDIT SCORE MAPPING ────────────────────────────────────

def prob_to_score(prob: float) -> int:
    """Log-odds → 300–850 (same math as FICO/CIBIL)."""
    prob      = max(1e-6, min(1 - 1e-6, prob))
    log_odds  = math.log(prob / (1 - prob))
    normalised= max(0.0, min(1.0, 1 - (log_odds + 4) / 8))
    return int(300 + normalised * 550)


def timing_to_pts(avg_days_early: float, repay_consistency: float) -> tuple[int, int, int]:
    """Return (early_pts, late_pts, net) based on your repayment timing rules."""
    early_pts = (30 if avg_days_early >= 10 else
                 20 if avg_days_early >= 5  else
                 10 if avg_days_early >= 1  else 0)
    max_late  = abs(min(avg_days_early, 0))
    late_pts  = (-70 if max_late >= 30 else
                 -35 if max_late >= 8  else
                 -15 if max_late >= 1  else 0)
    # Consistency boost: reward high consistency
    cons_bonus = int((repay_consistency - 0.5) * 20) if repay_consistency > 0.5 else 0
    return early_pts, late_pts, early_pts + late_pts + cons_bonus


def risk_band(score: int) -> dict:
    if   score >= 750: return {"label":"Excellent","color":"#1D9E75","decision":"Approve — Premium rates",   "band":5}
    elif score >= 700: return {"label":"Good",     "color":"#4CAF50","decision":"Approve — Standard rates",  "band":4}
    elif score >= 650: return {"label":"Fair",     "color":"#BA7517","decision":"Approve — Higher interest", "band":3}
    elif score >= 580: return {"label":"Poor",     "color":"#E67E22","decision":"Manual Review Required",    "band":2}
    else:              return {"label":"Very Poor","color":"#E24B4A","decision":"Decline — Too risky",       "band":1}


# ── 8. FULL PIPELINE RUN ───────────────────────────────────────

def run_pipeline(data_path: str = DATA_PATH) -> dict:
    print("\n" + "═"*52)
    print("  ML PIPELINE — German Credit Dataset")
    print("═"*52)

    print("\n[1] Loading data...")
    df_raw = load_data(data_path)

    print("\n[2] Encoding categoricals...")
    df_enc, encoders = encode(df_raw)

    print("\n[3] Adding repayment timing features...")
    df_full = add_timing_features(df_enc)

    print("\n[4] Splitting & scaling...")
    X_train, X_test, y_train, y_test, scaler = split_scale(df_full)

    print("\n[5] Training Random Forest (300 trees)...")
    rf = train_rf(X_train, y_train)

    print("\n[6] Training Logistic Regression (baseline)...")
    lr = train_lr(X_train, y_train)

    print("\n[7] Evaluation on test set...")
    rf_metrics = evaluate(rf, X_test, y_test, "Random Forest")
    lr_metrics = evaluate(lr, X_test, y_test, "Logistic Regression")

    print("\n[8] 5-Fold Cross-Validation...")
    rf_cv = cross_val(rf, X_train, y_train, "Random Forest")
    lr_cv = cross_val(lr, X_train, y_train, "Logistic Regression")

    print("\n[9] Feature importances...")
    importances = {
        col: round(float(v), 6)
        for col, v in sorted(
            zip(FEATURE_COLS, rf.feature_importances_),
            key=lambda x: -x[1]
        )
    }
    print("  Top 5:", list(importances.items())[:5])

    # Build metadata payload
    meta = {
        "model_type":       "RandomForestClassifier",
        "n_estimators":     300,
        "training_rows":    len(X_train),
        "test_rows":        len(X_test),
        "feature_count":    len(FEATURE_COLS),
        "features":         FEATURE_COLS,
        "dataset":          "German Credit Dataset (UCI)",
        "random_forest": {**rf_metrics, "cv": rf_cv},
        "logistic_regression": {**lr_metrics, "cv": lr_cv},
        "feature_importances": importances,
    }

    print("\n[10] Saving model artifacts...")
    joblib.dump(rf,       RF_PATH);     print(f"  ✓ {RF_PATH}")
    joblib.dump(lr,       LR_PATH);     print(f"  ✓ {LR_PATH}")
    joblib.dump(scaler,   SCALER_PATH); print(f"  ✓ {SCALER_PATH}")
    joblib.dump(encoders, ENC_PATH);    print(f"  ✓ {ENC_PATH}")
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ {META_PATH}")

    print("\n" + "═"*52)
    print(f"  Pipeline complete. RF AUC = {rf_metrics['auc']:.4f}")
    print("═"*52 + "\n")
    return meta


# ── 9. INFERENCE ───────────────────────────────────────────────
#   Called by Flask at prediction time.

_rf      = None
_scaler  = None
_enc     = None
_meta    = None


def load_artifacts():
    global _rf, _scaler, _enc, _meta
    _rf     = joblib.load(RF_PATH)
    _scaler = joblib.load(SCALER_PATH)
    _enc    = joblib.load(ENC_PATH)
    with open(META_PATH) as f:
        _meta = json.load(f)
    print("  Model artifacts loaded.")


def artifacts_exist() -> bool:
    return all(os.path.exists(p) for p in [RF_PATH, SCALER_PATH, ENC_PATH, META_PATH])


def _encode_input(raw: dict) -> pd.DataFrame:
    """
    Encode a raw applicant dict into a scaled feature vector.

    raw must contain:
      Categorical (string codes):
        checking_account, credit_history, purpose, savings_account,
        employment_since, personal_status, other_debtors, property,
        other_plans, housing, job, telephone, foreign_worker
      Numeric (numbers):
        duration, credit_amount, installment_rate, residence_since,
        age, existing_credits, people_liable
      Timing (numbers — from sliders):
        avg_days_early, repay_consistency
    """
    row = {}

    # Categorical — encode with saved LabelEncoders
    for col in CATEGORICAL_COLS:
        val = str(raw.get(col, "A14"))  # fallback to most common code
        le  = _enc[col]
        # handle unseen labels gracefully
        if val in le.classes_:
            row[col] = int(le.transform([val])[0])
        else:
            row[col] = 0

    # Numeric — pass through directly
    defaults = {
        "duration": 24, "credit_amount": 5000, "installment_rate": 2,
        "residence_since": 2, "age": 35, "existing_credits": 1, "people_liable": 1,
    }
    for col in NUMERIC_COLS:
        row[col] = float(raw.get(col, defaults.get(col, 0)))

    # Timing features — derived from user sliders
    avg_early   = float(raw.get("avg_days_early", 0))
    consistency = float(raw.get("repay_consistency", 0.8))
    ep, lp, net = timing_to_pts(avg_early, consistency)

    row["avg_days_early"]    = avg_early
    row["repay_consistency"] = consistency
    row["early_pay_score"]   = ep
    row["delay_penalty"]     = lp
    row["net_timing_score"]  = net

    df_row = pd.DataFrame([row])[FEATURE_COLS]
    scaled = pd.DataFrame(
        _scaler.transform(df_row),
        columns=FEATURE_COLS
    )
    return scaled, ep, lp, net


def predict_one(raw: dict) -> dict:
    """Score a single applicant. Returns full assessment dict."""
    scaled, ep, lp, net = _encode_input(raw)

    prob_default = float(_rf.predict_proba(scaled)[0][1])
    base_score   = prob_to_score(prob_default)
    final_score  = max(300, min(850, base_score + net))
    band         = risk_band(final_score)

    # SHAP-style factor analysis using feature importances
    fi  = _meta["feature_importances"]
    raw_vals = scaled.iloc[0].to_dict()  # scaled values for sign check
    factors = []
    label_map = {
        "credit_amount":    "Loan Amount",
        "duration":         "Loan Duration",
        "avg_days_early":   "Payment Timing",
        "repay_consistency":"Pay Consistency",
        "savings_account":  "Savings Account",
        "employment_since": "Employment Status",
        "checking_account": "Checking Account",
        "credit_history":   "Credit History",
        "age":              "Applicant Age",
        "installment_rate": "Instalment Rate",
        "net_timing_score": "Net Timing Score",
        "early_pay_score":  "Early Payment Bonus",
        "delay_penalty":    "Late Payment Penalty",
    }
    positive_high = {"repay_consistency","savings_account","employment_since",
                     "age","early_pay_score","net_timing_score"}
    for col, imp in fi.items():
        if col not in label_map:
            continue
        sv      = raw_vals.get(col, 0)
        is_pos  = (sv >= 0) if col in positive_high else (sv <= 0)
        factors.append({
            "feature":    label_map[col],
            "importance": round(imp * 100, 2),
            "impact":     "positive" if is_pos else "negative",
            "raw_value":  round(sv, 3),
        })

    factors.sort(key=lambda x: -x["importance"])

    return {
        "success":             True,
        "probability_default": round(prob_default, 4),
        "base_score":          base_score,
        "timing_adjustment":   net,
        "final_score":         final_score,
        "risk_label":          band["label"],
        "risk_color":          band["color"],
        "decision":            band["decision"],
        "risk_band":           band["band"],
        "score_breakdown": {
            "model_score":  base_score,
            "early_bonus":  ep,
            "late_penalty": lp,
            "final":        final_score,
        },
        "top_factors": factors[:8],
    }


def get_meta() -> dict:
    return _meta or {}


# ── Run standalone ─────────────────────────────────────────────
if __name__ == "__main__":
    import shutil
    # Copy dataset from parent directory if needed
    if not os.path.exists(DATA_PATH):
        parent = os.path.join(os.path.dirname(BASE_DIR), "german.data")
        if os.path.exists(parent):
            shutil.copy(parent, DATA_PATH)
            print(f"Copied dataset → {DATA_PATH}")
        else:
            print(f"ERROR: german.data not found at {DATA_PATH}")
            exit(1)
    run_pipeline()
