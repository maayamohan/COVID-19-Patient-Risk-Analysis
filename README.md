# COVID-19 Patient Risk Analysis

This project explores whether demographic and routinely recorded clinical features can rank COVID-19 cases by ICU admission, intubation, or recorded death. It adapts the triage question in Zhan et al. to the public Mexican COVID-19 surveillance dataset in this repository. It does not reproduce the CT/radiomics study and it is not a clinical decision tool.

## Dataset and targets

`dataset/Covid Data.csv` contains 1,048,575 rows and 21 columns. The target records are `ICU` and `INTUBED` (1 = outcome, 2 = no recorded outcome; all other codes are unknown and excluded) and `DATE_DIED` (`9999-99-99` = no recorded death; valid `dd/mm/yyyy` date = death). ICU and intubation labels occur only for `PATIENT_TYPE=2`, so those tasks cover hospitalized cases. Death dates are parsed strictly; malformed values are excluded.

The feature groups are Demo (`AGE`, `SEX`), DemoComorb (Demo plus diabetes, COPD, asthma, immunosuppression, hypertension, other disease, cardiovascular disease, obesity, chronic renal disease, and tobacco), and Clinical (DemoComorb plus `PNEUMONIA`). Outcome fields, `PATIENT_TYPE`, pregnancy, classification, and service/unit variables are excluded. Age codes 97/98/99, comorbidity codes 98/99, and pneumonia code 99 are treated as missing and imputed within the training pipeline.

The dataset has no patient identifier. It contains 812,049 exact duplicate full rows. The runner uses stratified grouped splitting on the full 13-feature candidate matrix, so feature subsets share the same held-out rows and identical candidate predictor rows stay on one side of the split. This reduces exact-feature-row leakage but cannot establish patient-level independence. Historical CSVs were produced with ordinary random splits and must be treated as exploratory, potentially optimistic results.

## Methods

Models: Logistic Regression, Random Forest, Linear SVM, MLP, and LightGBM. Feature-engineering modes: raw preprocessing, SMOTEENN, PCA (95% variance), L1 Logistic Regression feature selection, mutual-information SelectKBest (GUS-inspired proxy), and F-test SelectFpr. SMOTEENN is applied to training data only; the default cap is 100,000 training rows. The historical run's 3 failures are preserved in `results/smoteenn_failures.csv`.

Reported measures include accuracy, precision, recall, F1, ROC-AUC, average precision (called PR-AUC in the CSVs), and confusion-matrix counts. Because outcomes are imbalanced, accuracy is not used alone. No threshold tuning is performed on the held-out test set.

## Environment

From the repository root, in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the target/data summary or tests:

```powershell
python src/preprocessing_clean.py
python -m pytest -q
```

Run a small reproducible experiment (one model, one feature group, one target):

```powershell
python src/run_experiments.py --engineering raw --targets DEATH --feature-sets Clinical --models LogisticRegression
```

The group-aware runner writes to `results/raw_grouped_results.csv`, leaving historical row-random results untouched. Other modes use matching names such as `pca_grouped_results.csv`. Resume a grouped run with:

```powershell
python src/run_experiments.py --engineering lasso --targets DEATH --feature-sets Clinical --models LogisticRegression --resume
```

Run all 45 combinations for one feature-engineering mode with:

```powershell
python src/run_experiments.py --engineering raw
```

The raw, PCA, LASSO, and SMOTEENN CSVs already in `results/` are historical random-split results. They are not overwritten by grouped runs. GUS and FPR can be invoked with `--engineering gus` and `--engineering fpr`. Failed combinations append to a separate failure CSV.

To rebuild consolidated CSV summaries, the two-page report, and the slide deck from the saved results, run:

```powershell
python tools/build_deliverables.py
```

## Current saved experiment coverage

The historical files contain raw 45/45, Clinical PCA 15/15, LASSO 15/15 for Clinical at `C=1.0`, and SMOTEENN at 42/45. A newer group-aware Clinical Logistic Regression comparison covers all six methods (18 runs: 3 targets x 6 methods). GUS/FPR have three grouped runs each, but no historical rows. See `docs/experiment_results.md` for coverage, metrics, and caveats. Historical metrics are not leakage-controlled estimates because the original split allowed exact duplicate records on both sides.

Generated comparisons are embedded in the report (grouped-split F1 chart, page 2) and deck (historical best-F1 chart, slide 7). The underlying data are in `results/consolidated_results.csv` and the comparison-by-group CSV files.

## Limitations and responsible use

The source has no patient IDs, contains many repeated rows, has high outcome imbalance, and the public surveillance fields may be recorded after clinical presentation. ICU/intubation analyses apply only to cases with known labels, which are hospitalized cases. The data cannot demonstrate clinical validity, causal effects, transportability, or fairness. Do not use these models for patient care or resource allocation.

## Repository map

- `src/`: preprocessing, feature engineering, models, experiment runner, analysis scripts.
- `dataset/`: included source CSV.
- `results/`: historical experiment outputs and regenerated consolidated summaries.
- `reports/`: report source and PDF.
- `presentation/`: review deck.
- `docs/`: methodology, results audit, demo steps, and viva questions.
- `tests/`: focused synthetic pipeline checks.

## References

Zhan et al., “CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients.” The assigned study motivated the outcomes. The project uses an approved public tabular dataset and a different predictor set, so results do not reproduce that paper.
