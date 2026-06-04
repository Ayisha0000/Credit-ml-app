"""
app.py — Flask REST API for CreditIQ
Serves the trained German Credit RF model via HTTP endpoints.

Start:
    python app.py

Endpoints:
    GET  /api/health          — liveness check
    GET  /api/model-info      — metrics, feature importances, model metadata
    POST /api/predict         — score one applicant (JSON body)
    GET  /api/score-bands     — score band definitions
    GET  /api/feature-guide   — valid categorical codes for all fields
"""

import os, json, shutil
from flask import Flask, request, jsonify
from flask import Flask, render_template
from flask_cors import CORS

# ── Local ML module ────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ml_pipeline as ml

app = Flask(__name__)
CORS(app)  # allow all origins so the HTML frontend can call freely


@app.route('/')
def home():
    return render_template('index.html')


# ── Bootstrap: train if no saved model, else load ─────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
ROOT_DATA_PATH = os.path.join(REPO_ROOT, "data", "german.data")
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "german.data")
DATA_PATH = ROOT_DATA_PATH if os.path.exists(ROOT_DATA_PATH) else LOCAL_DATA_PATH

def bootstrap():
    """On startup: train model if artifacts don't exist, else load saved ones."""
    if not os.path.exists(DATA_PATH):
        # Try to find german.data one level up (where previous work placed it)
        parent = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "german.data")
        if os.path.exists(parent):
            shutil.copy(parent, DATA_PATH)
            print(f"  Dataset copied from {parent}")
        else:
            raise FileNotFoundError(
                "german.data not found. Download from:\n"
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data\n"
                "and place it in the ml_backend/ folder."
            )

    if ml.artifacts_exist():
        print("\n[BOOT] Saved model found — loading artifacts...")
        ml.load_artifacts()
    else:
        print("\n[BOOT] No saved model — training from German Credit Dataset...")
        ml.run_pipeline(data_path=DATA_PATH)
        ml.load_artifacts()
    print("[BOOT] Ready.\n")


# ── Routes ─────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Liveness + readiness check."""
    meta = ml.get_meta()
    return jsonify({
        "status":      "ok",
        "model_ready": bool(meta),
        "model_type":  meta.get("model_type", ""),
        "dataset":     meta.get("dataset", ""),
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return full model metadata including metrics and feature importances."""
    meta = ml.get_meta()
    if not meta:
        return jsonify({"error": "Model not loaded"}), 503

    rf = meta.get("random_forest", {})
    lr = meta.get("logistic_regression", {})

    return jsonify({
        "model_type":    meta["model_type"],
        "n_estimators":  meta["n_estimators"],
        "dataset":       meta["dataset"],
        "training_rows": meta["training_rows"],
        "test_rows":     meta["test_rows"],
        "feature_count": meta["feature_count"],
        "features":      meta["features"],

        # Random Forest metrics
        "auc":       rf.get("auc"),
        "accuracy":  rf.get("acc"),
        "precision": rf.get("precision"),
        "recall":    rf.get("recall"),
        "f1":        rf.get("f1"),
        "cv_auc_mean": rf.get("cv", {}).get("mean"),
        "cv_auc_std":  rf.get("cv", {}).get("std"),
        "confusion_matrix": rf.get("confusion_matrix"),

        # Baseline comparison
        "baseline_lr": {
            "auc":      lr.get("auc"),
            "accuracy": lr.get("acc"),
        },

        # Feature importances sorted descending
        "feature_importances": meta["feature_importances"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Score one loan applicant.

    Request JSON body (all fields optional — defaults provided):
    {
      // Categorical codes (see /api/feature-guide for valid values)
      "checking_account":  "A12",   // A11 <0 | A12 0-200 | A13 >200 | A14 none
      "credit_history":    "A32",   // A30-A34
      "purpose":           "A43",   // A40-A410
      "savings_account":   "A61",   // A61-A65
      "employment_since":  "A73",   // A71-A75
      "personal_status":   "A93",   // A91-A94
      "other_debtors":     "A101",  // A101-A103
      "property":          "A121",  // A121-A124
      "other_plans":       "A143",  // A141-A143
      "housing":           "A152",  // A151-A153
      "job":               "A173",  // A171-A174
      "telephone":         "A192",  // A191-A192
      "foreign_worker":    "A201",  // A201-A202

      // Numeric
      "duration":          24,      // months
      "credit_amount":     5000,    // amount in DM/currency units
      "installment_rate":  2,       // 1-4 (% of income)
      "residence_since":   2,       // years 1-4
      "age":               35,
      "existing_credits":  1,       // 1-4
      "people_liable":     1,       // 1-2

      // Timing (YOUR INNOVATION — from sliders)
      "avg_days_early":    7.0,     // positive=early, negative=late
      "repay_consistency": 0.85,    // 0.0–1.0
    }

    Response:
    {
      "success": true,
      "probability_default": 0.1823,
      "base_score":          718,
      "timing_adjustment":   +20,
      "final_score":         738,
      "risk_label":          "Good",
      "risk_color":          "#4CAF50",
      "decision":            "Approve — Standard rates",
      "risk_band":           4,
      "score_breakdown":     { ... },
      "top_factors":         [ ... ]
    }
    """
    try:
        raw = request.get_json(force=True)
        if raw is None:
            return jsonify({"success": False, "error": "No JSON body"}), 400

        result = ml.predict_one(raw)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/score-bands", methods=["GET"])
def score_bands():
    """Return the score band definitions used for display."""
    return jsonify([
        {"label":"Very Poor","range":"300–579","min":300,"max":579,"color":"#E24B4A","description":"High default risk"},
        {"label":"Poor",     "range":"580–649","min":580,"max":649,"color":"#E67E22","description":"Below average creditworthiness"},
        {"label":"Fair",     "range":"650–699","min":650,"max":699,"color":"#BA7517","description":"Acceptable with conditions"},
        {"label":"Good",     "range":"700–749","min":700,"max":749,"color":"#4CAF50","description":"Reliable borrower"},
        {"label":"Excellent","range":"750–850","min":750,"max":850,"color":"#1D9E75","description":"Top-tier creditworthiness"},
    ])


@app.route("/api/feature-guide", methods=["GET"])
def feature_guide():
    """Return valid category codes for all categorical fields."""
    return jsonify({
        "checking_account": {
            "A11": "< 0 DM",
            "A12": "0 to 200 DM",
            "A13": ">= 200 DM",
            "A14": "No checking account",
        },
        "credit_history": {
            "A30": "No credits / all paid back duly",
            "A31": "All credits paid back duly",
            "A32": "Existing credits paid back so far",
            "A33": "Delay in past",
            "A34": "Critical account",
        },
        "purpose": {
            "A40": "Car (new)",     "A41": "Car (used)",
            "A42": "Furniture",     "A43": "Radio/TV",
            "A44": "Appliances",    "A45": "Repairs",
            "A46": "Education",     "A48": "Retraining",
            "A49": "Business",      "A410": "Other",
        },
        "savings_account": {
            "A61": "< 100 DM",      "A62": "100–500 DM",
            "A63": "500–1000 DM",   "A64": ">= 1000 DM",
            "A65": "Unknown / no savings account",
        },
        "employment_since": {
            "A71": "Unemployed",    "A72": "< 1 year",
            "A73": "1–4 years",     "A74": "4–7 years",
            "A75": ">= 7 years",
        },
        "personal_status": {
            "A91": "Male: divorced/separated",
            "A92": "Female: divorced/separated/married",
            "A93": "Male: single",
            "A94": "Male: married/widowed",
        },
        "other_debtors": {
            "A101": "None", "A102": "Co-applicant", "A103": "Guarantor",
        },
        "property": {
            "A121": "Real estate", "A122": "Savings / Life insurance",
            "A123": "Car / Other", "A124": "Unknown / No property",
        },
        "other_plans": {
            "A141": "Bank",    "A142": "Stores",    "A143": "None",
        },
        "housing": {
            "A151": "Rent", "A152": "Own", "A153": "Free",
        },
        "job": {
            "A171": "Unemployed / unskilled (non-resident)",
            "A172": "Unskilled (resident)",
            "A173": "Skilled employee / official",
            "A174": "Management / self-employed / highly qualified",
        },
        "telephone": {
            "A191": "None registered",
            "A192": "Yes, registered",
        },
        "foreign_worker": {
            "A201": "Yes", "A202": "No",
        },
        "timing_sliders": {
            "avg_days_early":    "Positive = paid early, Negative = paid late. Range: -30 to +20",
            "repay_consistency": "Fraction of payments made on time or early. Range: 0.0 to 1.0",
        }
    })


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    bootstrap()
    print("CreditIQ ML API running on http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
