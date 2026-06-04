# Credit IQ — Combined Project

This repository contains two related components for the Credit IQ project:

- `credit_scoring_project/` — reproducible training pipeline and research code.
- `credit-ml-app/`         — lightweight Flask app serving trained models.

This top-level README documents the overall layout, setup, and recommended cleanup.

**What this project does:**
- Trains repayment-aware credit scoring models on the German Credit Dataset.
- Exposes a small Flask API for scoring single applicants and serving a simple UI.

**Recommended final structure** (clean, maintainable):

```
Credit IQ/
├── data/                      # canonical dataset location (german.data)
├── models/                    # canonical model artifacts (rf, lr, scaler, encoders, meta)
├── credit_scoring_project/    # training + experiments (library code, notebooks)
├── credit-ml-app/             # production-like Flask app and frontend
├── tests/                     # project tests (unit / integration)
├── docs/                      # design notes, architecture, PDFs
├── pyproject.toml             # project metadata for the training package
├── requirements.txt           # consolidated requirements (optional)
└── README.md                  # this file
```

Why this layout?
- Keeps training/research code separate from the serving app.
- A single `models/` folder is the canonical artifact store used by the app and research code.

-- Duplication & outputs
- There are two places currently storing artifacts:
  - `credit_scoring_project/outputs/` — contains `model_rf.pkl`, `scaler.pkl`, and a PNG report.
  - `credit-ml-app/backend/model_store/` — contains `rf_model.pkl`, `lr_model.pkl`, `scaler.pkl`, `encoders.pkl`, `model_meta.json`.

Recommendation: consolidate to a single `models/` directory at repository root. I will not move or delete artifacts until you approve — moving is safe because the two stores are redundant, but the Flask app expects `credit-ml-app/backend/model_store/` paths. After approval I will:

- copy artifacts into `models/` and update code references (or keep a small shim) so both `credit_scoring_project` and `credit-ml-app` can load from the canonical location.
- remove `credit_scoring_project/outputs/` once `models/` contains the required files and code is adjusted.

-- How to setup the environment

1) Create and activate a Python 3.10+ virtual environment (example using venv):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
```

2) Install dependencies for the training package (from `credit_scoring_project`):

```powershell
pip install -r credit_scoring_project\requirements.txt
```

3) Install dependencies for the Flask app (from `credit-ml-app/backend`):

```powershell
pip install -r credit-ml-app\backend\requirements.txt
```

-- How to train / run the model (research pipeline)

- From `credit_scoring_project/` you can run the full pipeline:

```powershell
cd credit_scoring_project
python main.py
```

This will produce artifacts under `credit_scoring_project/outputs/` (existing behavior). To consolidate artifacts, approve the consolidation step and I'll move them to `models/`.

-- How to run the ML app (Flask)

- From `credit-ml-app/backend/` run:

```powershell
cd credit-ml-app\backend
python app.py
```

On first run the app will train the model (if artifacts missing) and populate `credit-ml-app/backend/model_store/`.

-- Running the project from scratch

1) Create and activate a Python 3.10+ virtual environment from the repository root:

```powershell
C:/Users/ayish/AppData/Local/Programs/Python/Python312/python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

3) Install the training dependencies:

```powershell
pip install -r credit_scoring_project\requirements.txt
```

4) Install the Flask app dependencies:

```powershell
pip install -r credit-ml-app\backend\requirements.txt
```

5) Confirm the dataset is available at `data\german.data`. If not, download it with:

```powershell
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data" -OutFile ".\data\german.data"
```

6) Train and run the model pipeline:

```powershell
python .\credit_scoring_project\main.py
```

7) Run the Flask app:

```powershell
cd credit-ml-app\backend
python app.py
```

8) Open the app in your browser:

```text
http://localhost:5000
```


