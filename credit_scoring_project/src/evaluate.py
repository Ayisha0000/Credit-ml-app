# =============================================================
# evaluate.py — Model evaluation, metrics, and visualisation
# =============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PALETTE, REPORT_PATH, DECISION_THRESHOLD
from src.score import probability_to_score


def evaluate_model(model, X_test, y_test, model_name: str = "Model",
                   threshold: float = DECISION_THRESHOLD) -> dict:
    """
    Compute and print all evaluation metrics for a trained model.

    Uses a tuned threshold (0.45) instead of default 0.5:
        In banking, approving a bad loan costs more than rejecting a good one.
        Lowering the threshold means we're more cautious — we flag more cases
        as risky (higher recall for bad credit class).

    Returns a dict of all metrics and predictions for plotting.
    """
    y_prob = model.predict_proba(X_test)[:, 1]        # probability of default
    y_pred = (y_prob >= threshold).astype(int)         # apply tuned threshold

    auc   = roc_auc_score(y_test, y_prob)
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'─'*42}")
    print(f"  {model_name}")
    print(f"{'─'*42}")
    print(f"  AUC-ROC   : {auc:.4f}   ← Most important metric for credit scoring")
    print(f"  Accuracy  : {acc:.4f}   ← Overall correct predictions")
    print(f"  Precision : {prec:.4f}   ← When we say 'bad', how often correct?")
    print(f"  Recall    : {rec:.4f}   ← Of all bad applicants, how many caught?")
    print(f"  F1 Score  : {f1:.4f}   ← Balanced precision-recall measure")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Good','Bad'])}")

    return dict(
        name=model_name, auc=auc, acc=acc, precision=prec,
        recall=rec, f1=f1, y_prob=y_prob, y_pred=y_pred
    )


def plot_full_report(rf_res: dict, lr_res: dict, rf_model,
                     X_test, y_test, feature_cols: list,
                     out: str = REPORT_PATH):
    """
    Generate a comprehensive 6-panel evaluation report and save as PNG.

    Panels:
        1. Feature Importance  — which inputs matter most
        2. Confusion Matrix    — correct vs incorrect classifications
        3. ROC Curve           — RF vs LR trade-off at all thresholds
        4. Precision-Recall    — performance on imbalanced classes
        5. Score Distribution  — how scores separate good vs bad applicants
        6. Metric Comparison   — bar chart RF vs LR across all metrics
    """
    C = PALETTE
    fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
    fig.suptitle(
        "Credit Scoring Model — Full Evaluation Report",
        fontsize=17, fontweight="bold", color="#2C2C2A", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Feature Importance ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    top12 = importances.nlargest(12).sort_values()
    bar_colors = [C["green"] if v > top12.median() else C["purple"] for v in top12]
    bars = ax1.barh(top12.index, top12.values, color=bar_colors, height=0.6, edgecolor="none")
    for bar, val in zip(bars, top12.values):
        ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax1.set_title("Feature Importance (Top 12)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Importance Score")
    ax1.set_facecolor(C["bg"])
    ax1.spines[["top","right"]].set_visible(False)

    # ── 2. Confusion Matrix ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_test, rf_res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax2,
                xticklabels=["Good","Bad"], yticklabels=["Good","Bad"],
                linewidths=0.5, linecolor="white", annot_kws={"size": 14})
    ax2.set_title("Confusion Matrix\n(Random Forest)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")

    # ── 3. ROC Curves ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for res, color, ls in [(rf_res, C["green"], "-"), (lr_res, C["purple"], "--")]:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax3.plot(fpr, tpr, color=color, lw=2, ls=ls,
                 label=f"{res['name']} AUC={res['auc']:.3f}")
    ax3.plot([0,1],[0,1],"k:", lw=1, alpha=0.5, label="Random (0.500)")
    ax3.set_title("ROC Curve", fontweight="bold", fontsize=12)
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.legend(fontsize=8); ax3.set_facecolor(C["bg"])
    ax3.spines[["top","right"]].set_visible(False)

    # ── 4. Precision-Recall ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for res, color, ls in [(rf_res, C["green"], "-"), (lr_res, C["purple"], "--")]:
        prec_c, rec_c, _ = precision_recall_curve(y_test, res["y_prob"])
        ax4.plot(rec_c, prec_c, color=color, lw=2, ls=ls, label=res["name"])
    ax4.axhline(y_test.mean(), color=C["gray"], ls=":", lw=1.5,
                label=f"Baseline ({y_test.mean():.2f})")
    ax4.set_title("Precision-Recall Curve", fontweight="bold", fontsize=12)
    ax4.set_xlabel("Recall"); ax4.set_ylabel("Precision")
    ax4.legend(fontsize=8); ax4.set_facecolor(C["bg"])
    ax4.spines[["top","right"]].set_visible(False)

    # ── 5. Credit Score Distribution ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    scores_good = [probability_to_score(p) for p,t in zip(rf_res["y_prob"], y_test) if t==0]
    scores_bad  = [probability_to_score(p) for p,t in zip(rf_res["y_prob"], y_test) if t==1]
    ax5.hist(scores_good, bins=20, alpha=0.7, color=C["green"],
             label="Good credit", edgecolor="white")
    ax5.hist(scores_bad,  bins=20, alpha=0.7, color=C["red"],
             label="Bad credit",  edgecolor="white")
    ax5.axvline(700, color=C["amber"], lw=2, ls="--", label="Approve ≥700")
    ax5.axvline(580, color=C["gray"],  lw=1.5, ls=":", label="Review ≥580")
    ax5.set_title("Credit Score Distribution", fontweight="bold", fontsize=12)
    ax5.set_xlabel("Credit Score"); ax5.set_ylabel("Count")
    ax5.legend(fontsize=8); ax5.set_facecolor(C["bg"])
    ax5.spines[["top","right"]].set_visible(False)

    # ── 6. Metrics Bar Chart ──────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    metric_keys = ["auc","acc","precision","recall","f1"]
    labels = ["AUC-ROC","Accuracy","Precision","Recall","F1 Score"]
    x = np.arange(len(metric_keys))
    w = 0.35
    rf_vals = [rf_res[m] for m in metric_keys]
    lr_vals = [lr_res[m] for m in metric_keys]
    b_rf = ax6.bar(x - w/2, rf_vals, w, label="Random Forest",      color=C["green"], edgecolor="white")
    b_lr = ax6.bar(x + w/2, lr_vals, w, label="Logistic Regression", color=C["purple"], edgecolor="white")
    for bar in list(b_rf) + list(b_lr):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax6.set_xticks(x); ax6.set_xticklabels(labels)
    ax6.set_ylim(0, 1.12)
    ax6.set_title("Model Performance Comparison — Random Forest vs Logistic Regression",
                  fontweight="bold", fontsize=12)
    ax6.legend(fontsize=10); ax6.set_facecolor(C["bg"])
    ax6.spines[["top","right"]].set_visible(False)

    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"\nReport saved → {out}")


if __name__ == "__main__":
    from src.preprocess import run_preprocessing
    from src.train import train_random_forest, train_logistic_regression
    X_train, X_test, y_train, y_test, feature_cols, scaler, _ = run_preprocessing()
    rf = train_random_forest(X_train, y_train)
    lr = train_logistic_regression(X_train, y_train)
    rf_res = evaluate_model(rf, X_test, y_test, "Random Forest")
    lr_res = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    plot_full_report(rf_res, lr_res, rf, X_test, y_test, feature_cols)
