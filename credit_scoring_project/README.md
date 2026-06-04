# Dynamic Credit Scoring System

Repayment-aware ML credit scoring with early & late payment intelligence.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (or place german.data in data/ folder)
wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data -P data/

# 3. Run everything
python main.py
```

## What it does

- Trains a Random Forest classifier on the German Credit Dataset
- Engineers repayment timing features (early payment reward, late payment penalty)
- Outputs a 300–850 credit score per applicant
- Generates a 6-panel visual evaluation report

## Key Innovation

Most models ask: **"Did they miss a payment?"**
This model asks: **"How early or late did they pay — every single month?"**

- Pays 10+ days early → **+30 points** (Strong Potential Payer)
- Pays 5–9 days early → **+20 points**
- Pays 1–7 days late  → **-5 points**
- Pays 30+ days late  → **-70 points**

## Project Structure

```
credit_scoring_project/
├── data/                    # Dataset files
├── src/
│   ├── config.py            # All settings
│   ├── preprocess.py        # Data loading & encoding
│   ├── features.py          # Repayment timing features
│   ├── train.py             # Model training
│   ├── evaluate.py          # Metrics & charts
│   ├── score.py             # Credit score mapping
│   └── predict.py           # Predict new applicants
├── outputs/                 # Saved model & report
├── main.py                  # Run full pipeline
└── requirements.txt
```

## Developer Notes

- Make `src/` a proper package for imports — `import src.config`.
- Run the full pipeline with `python main.py` or `python -m src` after installing dependencies.
- Tests: `pytest -q` (the repository contains a basic smoke test).

