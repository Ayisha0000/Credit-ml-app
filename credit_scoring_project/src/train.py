# =============================================================
# train.py — Train Random Forest and Logistic Regression models
# =============================================================

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF, MAX_FEATURES,
    CLASS_WEIGHT, LR_C, LR_MAX_ITER, RANDOM_STATE,
    MODEL_PATH, SCALER_PATH
)


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train the primary Random Forest model.

    Random Forest = 300 decision trees, each trained on a
    random subset of data and features, then voting together.

    class_weight='balanced' is critical here:
        The dataset has 700 good vs 300 bad applicants.
        Without balancing, the model would learn to always predict "good"
        and achieve 70% accuracy while being useless for detecting bad loans.
        Balanced weighting makes the model pay 2.3x more attention to bad cases.
    """
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,        # 300 trees
        max_depth=MAX_DEPTH,              # Each tree asks max 10 questions
        min_samples_leaf=MIN_SAMPLES_LEAF,# At least 2 samples per leaf
        max_features=MAX_FEATURES,        # Each tree sees sqrt(n_features)
        class_weight=CLASS_WEIGHT,        # Handle 70/30 imbalance
        random_state=RANDOM_STATE,        # Reproducible results
        n_jobs=-1,                        # Use all CPU cores
    )
    rf.fit(X_train, y_train)
    print(f"  Trees trained: {rf.n_estimators}")
    print(f"  Features used: {rf.n_features_in_}")
    return rf


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Train the baseline Logistic Regression model.

    Used as a comparison point — if Random Forest is not significantly
    better than LR, it means the features are doing the heavy lifting.

    C=0.5 means moderate L2 regularisation — prevents overfitting
    on the training data.
    """
    print("Training Logistic Regression (baseline)...")
    lr = LogisticRegression(
        C=LR_C,                           # Regularisation strength
        max_iter=LR_MAX_ITER,             # Max convergence iterations
        class_weight=CLASS_WEIGHT,        # Same balancing as RF
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    print("  Logistic Regression trained.")
    return lr


def cross_validate_model(model, X_train, y_train, model_name: str = "Model") -> np.ndarray:
    """
    5-fold Stratified Cross-Validation.

    Instead of evaluating on just one test set, this splits the
    training data into 5 parts, trains on 4, tests on 1 — repeated
    5 times. The final AUC is the average of all 5 rounds.

    This gives a more reliable performance estimate and reveals
    if the model is overfitting to a specific split.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    print(f"  {model_name} — 5-fold CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def save_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Save trained model and scaler to disk using joblib."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Model saved  → {model_path}")
    print(f"  Scaler saved → {scaler_path}")


def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load a previously saved model and scaler from disk."""
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


if __name__ == "__main__":
    from src.preprocess import run_preprocessing
    X_train, X_test, y_train, y_test, feature_cols, scaler, _ = run_preprocessing()
    rf = train_random_forest(X_train, y_train)
    lr = train_logistic_regression(X_train, y_train)
    print("\n── Cross-Validation ──")
    cross_validate_model(rf, X_train, y_train, "Random Forest")
    cross_validate_model(lr, X_train, y_train, "Logistic Regression")
    save_model(rf, scaler)
    print("\nTraining complete.")
