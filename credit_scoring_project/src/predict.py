# =============================================================
# predict.py — Predict credit score for a new loan applicant
# =============================================================

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.score import full_credit_assessment
from src.config import CATEGORICAL_COLS, NUMERIC_COLS


def predict_applicant(
    model,
    scaler,
    feature_cols: list,
    applicant: dict,
    net_timing_score: int = 0
) -> dict:
    """
    Predict the creditworthiness of a single new loan applicant.

    Args:
        model            : trained RandomForestClassifier
        scaler           : fitted StandardScaler (from training)
        feature_cols     : list of feature column names (order matters!)
        applicant        : dict of pre-encoded applicant features
        net_timing_score : from features.py (early reward + late penalty)
                           Pass 0 if no repayment history available.

    Returns:
        Full credit assessment dict including score, risk label, decision.

    Example:
        result = predict_applicant(rf, scaler, feature_cols, {
            "credit_amount": 5000,
            "duration": 24,
            "age": 35,
            ...
        })
        print(result["final_score"])   # e.g. 712
        print(result["decision"])      # "Low Risk — Likely Approved"
    """
    # Build a single-row DataFrame with correct column order
    df_new = pd.DataFrame([applicant])[feature_cols]

    # Scale using the training scaler (must use transform, NOT fit_transform)
    df_new_sc = pd.DataFrame(
        scaler.transform(df_new),
        columns=feature_cols
    )

    # Get probability of default from the model
    prob_default = model.predict_proba(df_new_sc)[0][1]

    # Convert to full credit assessment (score + label)
    result = full_credit_assessment(prob_default, net_timing_score)
    return result


def print_assessment(result: dict, applicant_name: str = "Applicant") -> None:
    """Pretty-print the credit assessment result."""
    print(f"\n{'═'*45}")
    print(f"  Credit Assessment: {applicant_name}")
    print(f"{'═'*45}")
    print(f"  Probability of Default : {result['probability_of_default']:.4f} ({result['probability_of_default']*100:.1f}%)")
    print(f"  Base Score (model)     : {result['base_score']}")
    print(f"  Timing Adjustment      : {result['timing_adjustment']:+d} pts")
    print(f"  ──────────────────────────────────────")
    print(f"  FINAL CREDIT SCORE     : {result['final_score']}")
    print(f"  Risk Category          : {result['risk_category']}")
    print(f"  Decision               : {result['decision']}")
    print(f"{'═'*45}\n")


if __name__ == "__main__":
    from src.preprocess import run_preprocessing
    from src.train import train_random_forest

    # Load data and train
    X_train, X_test, y_train, y_test, feature_cols, scaler, _ = run_preprocessing()
    rf = train_random_forest(X_train, y_train)

    # ── Example 1: Good applicant (likely to be approved) ──
    good_applicant = {col: 0 for col in feature_cols}
    good_applicant.update({
        "credit_amount"   : 3000,    # Reasonable loan amount
        "duration"        : 12,      # Short loan term
        "age"             : 45,      # Mature, financially stable
        "installment_rate": 1,       # Low repayment burden
        "residence_since" : 4,       # Long-time resident
        "existing_credits": 1,       # Not over-borrowed
        "people_liable"   : 1,
    })
    result1 = predict_applicant(
        rf, scaler, feature_cols,
        good_applicant,
        net_timing_score=+20    # Pays early — Good Potential Payer
    )
    print_assessment(result1, "Schwartz (Early payer, small loan)")

    # ── Example 2: Risky applicant (likely to be declined) ──
    risky_applicant = {col: 0 for col in feature_cols}
    risky_applicant.update({
        "credit_amount"   : 15000,   # Very large loan
        "duration"        : 60,      # Long repayment period
        "age"             : 22,      # Young, limited history
        "installment_rate": 4,       # High burden on income
        "residence_since" : 1,
        "existing_credits": 3,       # Multiple active loans
        "people_liable"   : 2,
    })
    result2 = predict_applicant(
        rf, scaler, feature_cols,
        risky_applicant,
        net_timing_score=-35    # Pays late — chronic delay
    )
    print_assessment(result2, "Applicant B (Late payer, large loan)")

    # ── Example 3: Borderline applicant ───────────────────────
    border_applicant = {col: 0 for col in feature_cols}
    border_applicant.update({
        "credit_amount"   : 7000,
        "duration"        : 36,
        "age"             : 33,
        "installment_rate": 2,
        "residence_since" : 2,
        "existing_credits": 2,
        "people_liable"   : 1,
    })
    result3 = predict_applicant(
        rf, scaler, feature_cols,
        border_applicant,
        net_timing_score=0    # Pays exactly on deadline
    )
    print_assessment(result3, "Applicant C (On-time payer, medium loan)")
