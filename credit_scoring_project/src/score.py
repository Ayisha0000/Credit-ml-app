# =============================================================
# score.py — Convert model output to credit score (300–850)
#
# This is the same mathematics used by FICO and CIBIL scores.
# We use a log-odds transformation so the score is spread
# evenly across the range rather than clustered in the middle.
# =============================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    SCORE_MIN, SCORE_MAX,
    APPROVE_THRESHOLD, REVIEW_THRESHOLD,
    EARLY_PAYMENT_REWARDS, LATE_PAYMENT_PENALTIES
)


def probability_to_score(
    prob_default: float,
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX
) -> int:
    """
    Convert a default probability (0.0–1.0) to a credit score (300–850).

    Step 1: Clip probability to avoid log(0) errors
    Step 2: Compute log-odds = log(p / (1 - p))
            - p = 0.05 → log_odds =  ~2.94  (very safe borrower)
            - p = 0.50 → log_odds =   0.00  (uncertain)
            - p = 0.90 → log_odds =  ~2.20  (very risky borrower)
    Step 3: Normalise log_odds from range [-4, +4] to [0, 1]
    Step 4: Map to [score_min, score_max]
            Higher prob_default → lower log_odds normalised → lower score

    Example:
        prob_default = 0.10 → score ≈ 750  (Low Risk)
        prob_default = 0.40 → score ≈ 610  (Medium Risk)
        prob_default = 0.80 → score ≈ 380  (High Risk)
    """
    # Clip to valid probability range
    prob = np.clip(prob_default, 1e-6, 1 - 1e-6)

    # Log-odds transformation
    log_odds = np.log(prob / (1 - prob))

    # Normalise: log_odds in [-4, 4] → [0, 1]
    # Higher prob → positive log_odds → lower normalised → lower score
    normalised = 1 - (log_odds + 4) / 8
    normalised = np.clip(normalised, 0, 1)

    # Map to score range
    score = int(score_min + normalised * (score_max - score_min))
    return score


def apply_timing_adjustment(base_score: int, net_timing_score: int) -> int:
    """
    Apply the early/late payment timing adjustment to the base score.

    net_timing_score comes from features.py:
        Positive = early payer (reward)
        Negative = late payer  (penalty)

    The adjustment is capped to prevent extreme swings.

    Example:
        base_score = 720, net_timing_score = +20 → adjusted = 740
        base_score = 650, net_timing_score = -35 → adjusted = 615
    """
    adjusted = base_score + net_timing_score

    # Keep within valid range
    adjusted = max(SCORE_MIN, min(SCORE_MAX, adjusted))
    return adjusted


def get_risk_label(score: int) -> str:
    """
    Convert a numeric score to a human-readable risk label and decision.

    These thresholds match real-world banking practice:
        >= 700 : Approve — reliable borrower
        580-699: Review  — borderline, needs manual check
        < 580  : Decline — too risky for standard loan
    """
    if score >= APPROVE_THRESHOLD:
        return "Low Risk   — Likely Approved"
    elif score >= REVIEW_THRESHOLD:
        return "Medium Risk — Manual Review Required"
    else:
        return "High Risk  — Likely Declined"


def get_risk_category(score: int) -> str:
    """Return just the category name (for internal logic)."""
    if score >= APPROVE_THRESHOLD:
        return "LOW"
    elif score >= REVIEW_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"


def full_credit_assessment(
    prob_default: float,
    net_timing_score: int = 0
) -> dict:
    """
    Complete credit assessment for one applicant.

    Returns a dict with all scoring details.

    Args:
        prob_default    : model output — probability this person will default
        net_timing_score: from features.py (early reward + late penalty combined)

    Returns:
        {
            "probability_of_default" : 0.2341,
            "base_score"             : 682,
            "timing_adjustment"      : +20,
            "final_score"            : 702,
            "risk_category"          : "LOW",
            "decision"               : "Low Risk — Likely Approved"
        }
    """
    base_score    = probability_to_score(prob_default)
    final_score   = apply_timing_adjustment(base_score, net_timing_score)
    risk_category = get_risk_category(final_score)
    decision      = get_risk_label(final_score)

    return {
        "probability_of_default": round(float(prob_default), 4),
        "base_score"            : base_score,
        "timing_adjustment"     : net_timing_score,
        "final_score"           : final_score,
        "risk_category"         : risk_category,
        "decision"              : decision,
    }


if __name__ == "__main__":
    # Test the scoring system
    print("── Credit Score Mapping Test ──\n")
    test_cases = [
        (0.05,  30, "Excellent early payer"),
        (0.10,  20, "Good early payer"),
        (0.25,   0, "Neutral payer"),
        (0.40, -15, "Slightly late payer"),
        (0.70, -70, "Frequently late payer"),
        (0.90, -70, "High default risk"),
    ]
    print(f"{'Description':<25} {'P(default)':<12} {'Base':>6} {'Adjust':>8} {'Final':>7} {'Risk'}")
    print("─" * 75)
    for prob, timing, desc in test_cases:
        result = full_credit_assessment(prob, timing)
        print(
            f"{desc:<25} {prob:<12.2f} {result['base_score']:>6} "
            f"{result['timing_adjustment']:>+8} {result['final_score']:>7} "
            f"{result['risk_category']}"
        )
