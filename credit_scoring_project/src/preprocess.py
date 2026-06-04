# =============================================================
# preprocess.py — Load, clean, encode, and split the dataset
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_PATH, COLUMNS, CATEGORICAL_COLS, NUMERIC_COLS,
    TEST_SIZE, RANDOM_STATE
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the German Credit Dataset.

    The file is space-separated with no header row.
    Target column: 1 = good credit, 2 = bad credit
    We remap to: 0 = good (will repay), 1 = bad (will default)
    This makes 1 = "positive class" = the risky case we want to detect.
    """
    df = pd.read_csv(path, sep=" ", names=COLUMNS)

    # Recode target — 1=good→0, 2=bad→1
    df["target"] = df["target"].map({1: 0, 2: 1})

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Class distribution:")
    counts = df["target"].value_counts().rename({0: "Good credit", 1: "Bad credit"})
    for label, count in counts.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    print()
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """Print basic data quality information."""
    print("── Data Quality Check ──")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("  No missing values found.")
    else:
        print(f"  Missing values:\n{null_counts[null_counts > 0]}")
    print(f"  Duplicate rows: {df.duplicated().sum()}")
    print()


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all categorical columns.

    LabelEncoder converts text categories like 'A11', 'A12' into
    integers 0, 1, 2... so the ML model can process them.

    Returns the encoded DataFrame and a dict of encoders
    (needed to encode new applicants at prediction time).
    """
    data = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le  # save for later use

    return data, encoders


def split_and_scale(df: pd.DataFrame) -> tuple:
    """
    1. Separate features (X) from target (y)
    2. Split into 80% train / 20% test — stratified to preserve class ratio
    3. Apply StandardScaler — transforms all features to mean=0, std=1

    Scaling is required for Logistic Regression and ensures consistency
    across all models. Random Forest doesn't need it but it doesn't hurt.

    Returns: X_train, X_test, y_train, y_test, feature_cols, scaler
    """
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[feature_cols]
    y = df["target"]

    # Stratified split — preserves 70/30 class ratio in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),  # fit on train only — prevents data leakage
        columns=feature_cols,
        index=X_train.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),        # transform test using train's statistics
        columns=feature_cols,
        index=X_test.index
    )

    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Train class ratio — Good: {(y_train==0).sum()}  Bad: {(y_train==1).sum()}")
    print(f"Test  class ratio — Good: {(y_test==0).sum()}  Bad: {(y_test==1).sum()}\n")

    return X_train_sc, X_test_sc, y_train, y_test, feature_cols, scaler


def run_preprocessing(path: str = DATA_PATH) -> tuple:
    """Run the full preprocessing pipeline and return everything needed for training."""
    df_raw = load_data(path)
    check_data_quality(df_raw)
    df_encoded, encoders = encode_categoricals(df_raw)
    X_train, X_test, y_train, y_test, feature_cols, scaler = split_and_scale(df_encoded)
    return X_train, X_test, y_train, y_test, feature_cols, scaler, encoders


if __name__ == "__main__":
    run_preprocessing()
    print("Preprocessing complete.")
