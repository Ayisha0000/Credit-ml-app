#!/usr/bin/env python3
# =============================================================
# main.py — Run the complete Credit Scoring Pipeline
#
# USAGE:
#     python main.py
#
# This single command will:
#   1. Load and preprocess the German Credit Dataset
#   2. Generate synthetic repayment timing features
#   3. Train Random Forest and Logistic Regression
#   4. Evaluate both models (AUC, accuracy, precision, recall)
#   5. Run 5-fold cross-validation
#   6. Generate the 6-panel evaluation report (PNG)
#   7. Show credit score predictions for 3 example applicants
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import run_preprocessing
from src.features import generate_sample_repayment_data, compute_repayment_features
from src.train import train_random_forest, train_logistic_regression, cross_validate_model, save_model
from src.evaluate import evaluate_model, plot_full_report
from src.predict import predict_applicant, print_assessment
from src.score import full_credit_assessment


def print_banner(text: str) -> None:
    print(f"\n{'═'*50}")
    print(f"  {text}")
    print(f"{'═'*50}\n")


def main():
    print_banner("DYNAMIC CREDIT SCORING SYSTEM")
    print("Dataset     : German Credit Dataset (UCI)")
    print("Primary Model: Random Forest Classifier")
    print("Innovation  : Early & Late Payment Timing Scoring\n")

    # ── Step 1: Preprocess ──────────────────────────────────
    print_banner("Step 1 — Data Preprocessing")
    X_train, X_test, y_train, y_test, feature_cols, scaler, encoders = run_preprocessing()

    # ── Step 2: Repayment Features ──────────────────────────
    print_banner("Step 2 — Repayment Timing Features (Innovation)")
    os.makedirs("data", exist_ok=True)
    repay_df = generate_sample_repayment_data(
        n_applicants=200, months=12,
        output_path="data/sample_repayment.csv"
    )
    timing_features = compute_repayment_features(repay_df)
    print(f"\nSample timing features computed for {len(timing_features)} applicants")
    print(timing_features[["applicant_id","avg_days_before_due",
                             "repayment_consistency","early_payment_score",
                             "delay_penalty_score","net_timing_score"]].head(6).to_string(index=False))

    # ── Step 3: Train Models ─────────────────────────────────
    print_banner("Step 3 — Model Training")
    rf = train_random_forest(X_train, y_train)
    lr = train_logistic_regression(X_train, y_train)

    # ── Step 4: Evaluate ─────────────────────────────────────
    print_banner("Step 4 — Model Evaluation")
    rf_res = evaluate_model(rf, X_test, y_test, "Random Forest")
    lr_res = evaluate_model(lr, X_test, y_test, "Logistic Regression")

    # ── Step 5: Cross-Validation ─────────────────────────────
    print_banner("Step 5 — Cross-Validation (5-Fold)")
    cross_validate_model(rf, X_train, y_train, "Random Forest")
    cross_validate_model(lr, X_train, y_train, "Logistic Regression")

    # ── Step 6: Save model ───────────────────────────────────
    print_banner("Step 6 — Save Model")
    save_model(rf, scaler)

    # ── Step 7: Generate Report ──────────────────────────────
    print_banner("Step 7 — Generate Visual Report")
    plot_full_report(rf_res, lr_res, rf, X_test, y_test, feature_cols)

    # ── Step 8: Example Predictions ─────────────────────────
    print_banner("Step 8 — Example Applicant Predictions")

    # Schwartz: pays early, small manageable loan
    schwartz = {col: 0 for col in feature_cols}
    schwartz.update({"credit_amount": 3000, "duration": 12, "age": 45,
                     "installment_rate": 1, "residence_since": 4,
                     "existing_credits": 1, "people_liable": 1})
    r1 = predict_applicant(rf, scaler, feature_cols, schwartz, net_timing_score=+20)
    print_assessment(r1, "Schwartz — Pays 8 days early, small loan")

    # Risky applicant
    risky = {col: 0 for col in feature_cols}
    risky.update({"credit_amount": 15000, "duration": 60, "age": 22,
                  "installment_rate": 4, "residence_since": 1,
                  "existing_credits": 3, "people_liable": 2})
    r2 = predict_applicant(rf, scaler, feature_cols, risky, net_timing_score=-35)
    print_assessment(r2, "Applicant B — Pays 3 weeks late, large loan")

    # Borderline
    border = {col: 0 for col in feature_cols}
    border.update({"credit_amount": 7000, "duration": 36, "age": 33,
                   "installment_rate": 2, "residence_since": 2,
                   "existing_credits": 2, "people_liable": 1})
    r3 = predict_applicant(rf, scaler, feature_cols, border, net_timing_score=0)
    print_assessment(r3, "Applicant C — Pays exactly on deadline")

    # ── Summary ──────────────────────────────────────────────
    print_banner("Run Complete — Summary")
    print(f"  Random Forest AUC-ROC : {rf_res['auc']:.4f}")
    print(f"  Random Forest Accuracy: {rf_res['acc']:.4f}")
    print(f"  LR Baseline AUC-ROC   : {lr_res['auc']:.4f}")
    print(f"\n  Outputs:")
    print(f"    outputs/credit_scoring_report.png  — Visual evaluation report")
    print(f"    outputs/model_rf.pkl               — Saved trained model")
    print(f"    outputs/scaler.pkl                 — Saved scaler")
    print(f"\n  Files:")
    for f in ["src/config.py","src/preprocess.py","src/features.py",
              "src/train.py","src/evaluate.py","src/score.py","src/predict.py"]:
        print(f"    {f}")
    print()


if __name__ == "__main__":
    main()
