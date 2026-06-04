# =============================================================
# features.py — Repayment Timing Feature Engineering
#
# THIS IS THE CORE INNOVATION OF YOUR PROJECT.
#
# Instead of only asking "Did they miss a payment?", we ask:
#   - HOW EARLY did they pay? (reward)
#   - HOW LATE did they pay? (graded penalty)
#   - HOW CONSISTENT are they overall? (consistency ratio)
#
# These features get added to the model as extra inputs,
# improving accuracy and making the score fairer.
# =============================================================

import pandas as pd
import numpy as np
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import EARLY_PAYMENT_REWARDS, LATE_PAYMENT_PENALTIES


# ── Early Payment Reward ──────────────────────────────────────

def compute_early_payment_score(avg_days_before_due: float) -> int:
    """
    Convert average days-before-due into a reward score.

    Positive avg_days = paid early (good)
    Negative avg_days = paid late  (bad — this function returns 0 for late)

    Example:
        avg_days = 8.0  → paid 8 days early on average → +20 points
        avg_days = -3.0 → paid late on average          →  0 points (penalty handled separately)
    """
    if avg_days_before_due >= 10:
        return 30   # Strong Potential Payer
    elif avg_days_before_due >= 5:
        return 20   # Good Potential Payer
    elif avg_days_before_due >= 1:
        return 10   # Responsible Payer
    else:
        return 0    # No early-payment reward


# ── Late Payment Penalty ──────────────────────────────────────

def compute_delay_penalty(max_days_late: float) -> int:
    """
    Convert worst delay (in days) into a penalty score.

    max_days_late is always positive (it is the magnitude of the worst delay).
    Returns a NEGATIVE number (penalty).

    Example:
        max_days_late = 45 → paid 45 days late at worst → -70 points
        max_days_late = 5  → paid 5 days late at worst  → -15 points
        max_days_late = 0  → never late                 →  0 points
    """
    if max_days_late == 0:
        return 0      # Perfect payer — no penalty
    elif max_days_late <= 7:
        return -5     # Minor slip — small penalty
    elif max_days_late <= 30:
        return -15    # Moderate delay
    elif max_days_late <= 60:
        return -35    # Serious delay
    else:
        return -70    # Severe delay / near default


# ── Payment Trend ─────────────────────────────────────────────

def compute_payment_trend(days_series: pd.Series) -> float:
    """
    Calculate if the person's payment timing is improving or worsening over time.

    Uses linear regression slope on (month_index, days_before_due):
      Positive slope = paying earlier over time     = good trend
      Negative slope = paying later over time       = bad trend
      Near zero      = consistent behaviour

    Returns the slope value (float).
    """
    if len(days_series) < 3:
        return 0.0   # Not enough data for a reliable trend

    x = np.arange(len(days_series))
    y = days_series.values
    slope, _, _, _, _ = stats.linregress(x, y)
    return round(float(slope), 4)


# ── Main Feature Engineering Function ────────────────────────

def compute_repayment_features(repayment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all repayment-timing features per applicant.

    INPUT DataFrame must have these columns:
        applicant_id  — unique ID matching the main credit dataset
        due_date      — date the payment was due (string or datetime)
        paid_date     — date the payment was actually made (string or datetime)

    OUTPUT: one row per applicant with these new features:
        avg_days_before_due     — average days early (positive) or late (negative)
        repayment_consistency   — fraction of payments made on time or early (0.0–1.0)
        max_days_late           — worst delay in days (0 if never late)
        early_payment_score     — reward points for early payment behaviour
        delay_penalty_score     — penalty points for late payment behaviour
        payment_trend           — slope of payment timing over time (+ve = improving)
        net_timing_score        — early_payment_score + delay_penalty_score combined

    Example usage:
        df_repay = pd.read_csv('data/sample_repayment.csv')
        timing_features = compute_repayment_features(df_repay)
    """
    # Parse dates robustly
    repayment_df = repayment_df.copy()
    repayment_df["due_date"]  = pd.to_datetime(repayment_df["due_date"])
    repayment_df["paid_date"] = pd.to_datetime(repayment_df["paid_date"])

    # days_diff: positive = paid early, negative = paid late
    # Example: due Jan 2, paid Dec 25 → (Jan2 - Dec25) = +8 days (early)
    # Example: due Jan 2, paid Jan 10 → (Jan2 - Jan10) = -8 days (late)
    repayment_df["days_diff"] = (
        repayment_df["due_date"] - repayment_df["paid_date"]
    ).dt.days

    # Group by applicant and compute aggregate features
    def agg_features(group):
        days = group["days_diff"]
        return pd.Series({
            # Average days before due (your core feature)
            "avg_days_before_due": round(days.mean(), 2),

            # Consistency: what % of payments were on time or early?
            "repayment_consistency": round((days >= 0).mean(), 4),

            # Worst delay ever (in days, always positive)
            "max_days_late": abs(min(days.min(), 0)),

            # Payment trend (improving = positive slope)
            "payment_trend": compute_payment_trend(days),
        })

    features = repayment_df.groupby("applicant_id").apply(agg_features).reset_index()

    # Apply your reward and penalty systems
    features["early_payment_score"] = features["avg_days_before_due"].apply(
        compute_early_payment_score
    )
    features["delay_penalty_score"] = features["max_days_late"].apply(
        compute_delay_penalty
    )

    # Net timing score = reward + penalty combined
    features["net_timing_score"] = (
        features["early_payment_score"] + features["delay_penalty_score"]
    )

    return features


# ── Synthetic Data Generator (for demo/testing) ───────────────

def generate_sample_repayment_data(
    n_applicants: int = 200,
    months: int = 12,
    output_path: str = "data/sample_repayment.csv"
) -> pd.DataFrame:
    """
    Generate a realistic synthetic repayment timeline dataset.

    Creates three types of borrowers:
        - Good payers    (60%): mostly pay early or on time
        - Mixed payers   (25%): sometimes early, sometimes late
        - Risky payers   (15%): mostly late, some misses

    This data is used alongside the German Credit Dataset to add
    the repayment timing features to the model.
    """
    np.random.seed(42)
    records = []

    for i in range(n_applicants):
        # Assign borrower type
        roll = np.random.random()
        if roll < 0.60:
            borrower_type = "good"      # Pays early most of the time
        elif roll < 0.85:
            borrower_type = "mixed"     # Inconsistent
        else:
            borrower_type = "risky"     # Mostly late

        for month in range(1, months + 1):
            # Simulate a due date (2nd of each month)
            due_day   = pd.Timestamp(f"2024-{month:02d}-02")
            
            if borrower_type == "good":
                # Pays 5–15 days early, occasionally exactly on time
                days_early = np.random.choice(
                    [15, 12, 10, 8, 7, 5, 3, 1, 0],
                    p=[0.10, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.03, 0.02]
                )
                paid_day = due_day - pd.Timedelta(days=int(days_early))

            elif borrower_type == "mixed":
                # Mix of early and slightly late
                days_offset = np.random.randint(-15, 10)
                paid_day = due_day - pd.Timedelta(days=int(days_offset))

            else:  # risky
                # Mostly late, sometimes significantly
                days_late = np.random.choice(
                    [0, 5, 15, 30, 45, 60],
                    p=[0.05, 0.20, 0.30, 0.25, 0.15, 0.05]
                )
                paid_day = due_day + pd.Timedelta(days=int(days_late))

            records.append({
                "applicant_id"  : i,
                "month"         : month,
                "due_date"      : due_day.strftime("%Y-%m-%d"),
                "paid_date"     : paid_day.strftime("%Y-%m-%d"),
                "borrower_type" : borrower_type,
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample repayment data saved: {output_path} ({len(df)} rows)")
    return df


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    # Generate sample repayment data
    repay_df = generate_sample_repayment_data(
        n_applicants=200,
        months=12,
        output_path="data/sample_repayment.csv"
    )

    # Compute features
    timing_features = compute_repayment_features(repay_df)

    print("\nSample repayment features (first 5 applicants):")
    print(timing_features.head().to_string(index=False))

    print("\nFeature summary:")
    print(timing_features.describe().round(2).to_string())
