# Live demonstration guide

## Setup (PowerShell)

From the repository directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again. The project can also run from an already prepared Python environment with the packages in `requirements.txt`.

## Short live run

```powershell
python -m pytest -q
python src/run_experiments.py --engineering raw --targets ICU --feature-sets Clinical --models LogisticRegression
```

This reads the included CSV, constructs the known-label ICU cohort, uses the group-aware split to keep identical predictor rows together, fits preprocessing only on training data, trains one model, and appends the metrics to `results/raw_grouped_results.csv`. The full CSV is large, so allow a few minutes on a student laptop.

For the quickest review, show the already saved historical outputs without retraining:

```powershell
python -c "import pandas as pd; x=pd.read_csv('results/raw_baseline_results.csv'); print(x.groupby('target')[['precision','recall','f1','roc_auc','pr_auc']].mean().round(3))"
```

Explain that historical metrics came from a random row split and can be optimistic because the source contains repeated rows. The new grouped run is in a separate file and does not replace them.

## What to point out

1. `src/preprocessing_clean.py` constructs the targets, translates unknown sentinels to missing values, and makes a grouped split.
2. `src/feature_engineering.py` keeps the feature groups and training pipeline together.
3. `src/run_experiments.py` saves a result row after each successful fit and logs failures separately.
4. `results/` preserves the historical outputs, the failure log, and newly grouped results.

## Common issues

- **Missing dependency:** install with `python -m pip install -r requirements.txt`.
- **Dataset not found:** run commands from the repository root; the loader resolves the dataset relative to the source file.
- **Slow SMOTEENN:** the default cap is 100,000 training rows. For a quick demo choose `--engineering raw`.
- **SMOTEENN failure:** the ENN cleaning can remove a class in small/imbalanced feature groups. The runner records that combination in the matching `*_failures.csv` file.
- **PowerShell activation denied:** set execution policy for the current process only, as above, or invoke `.venv\Scripts\python.exe` directly.
