# =============================================================
# config.py — Central configuration for Credit Scoring Project
# Change values here to experiment without touching other files
# =============================================================

import os

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "german.data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "model_rf.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "credit_scoring_report.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────
# Column names matching german.data format (space-separated, no header)
COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since",
    "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_plans",
    "housing", "existing_credits", "job", "people_liable",
    "telephone", "foreign_worker", "target",
]

# Columns that contain text categories (need encoding)
CATEGORICAL_COLS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status", "other_debtors", "property",
    "other_plans", "housing", "job", "telephone", "foreign_worker",
]

# Columns that are already numbers
NUMERIC_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "people_liable",
]

# ── Model Hyperparameters ────────────────────────────────────
TEST_SIZE    = 0.20   # 20% held out for testing
RANDOM_STATE = 42     # Fixed seed — ensures same results every run

# Random Forest settings
N_ESTIMATORS    = 300   # Number of trees — more = more stable
MAX_DEPTH       = 10    # Max questions per tree — prevents overfitting
MIN_SAMPLES_LEAF = 2    # Min samples per leaf — smoother boundaries
MAX_FEATURES    = "sqrt"# Each tree sees sqrt(n_features) — ensures diversity
CLASS_WEIGHT    = "balanced"  # Handles 70/30 class imbalance

# Logistic Regression settings
LR_C         = 0.5     # Regularisation strength (smaller = more regularised)
LR_MAX_ITER  = 1000    # Max iterations to converge

# Decision threshold (default 0.5 — tuned to 0.45 for banking context)
# Because false approvals cost more than false rejections in banking
DECISION_THRESHOLD = 0.45

# ── Credit Score Mapping ─────────────────────────────────────
SCORE_MIN = 300   # Minimum score (worst)
SCORE_MAX = 850   # Maximum score (best)

# Score thresholds for risk labels
APPROVE_THRESHOLD = 700   # Score >= 700 → Low Risk → Approve
REVIEW_THRESHOLD  = 580   # Score >= 580 → Medium Risk → Manual review
# Score < 580 → High Risk → Decline

# ── Early/Late Payment Rewards & Penalties ───────────────────
# These implement YOUR innovation — graded repayment timing scoring

EARLY_PAYMENT_REWARDS = {
    10: 30,   # 10+ days early → Strong Potential Payer → +30 pts
    5:  20,   # 5-9 days early → Good Potential Payer   → +20 pts
    1:  10,   # 1-4 days early → Responsible Payer      → +10 pts
    0:   0,   # On deadline    → Neutral                →  +0 pts
}

LATE_PAYMENT_PENALTIES = {
    30:  -70,   # 30+ days late → Severe   → -70 pts
    8:   -35,   # 8-29 days late → Serious → -35 pts
    1:   -15,   # 1-7 days late → Minor    → -15 pts
    0:    -5,   # Same day late  → Tiny     →  -5 pts
}

# ── Visualisation ─────────────────────────────────────────────
PALETTE = {
    "green"  : "#1D9E75",
    "d_green": "#0F6E56",
    "purple" : "#7F77DD",
    "amber"  : "#BA7517",
    "red"    : "#E24B4A",
    "gray"   : "#888780",
    "bg"     : "#F8F8F6",
    "light"  : "#F4F4F2",
}
