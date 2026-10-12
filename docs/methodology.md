# Methodology and data audit

## Dataset audit

The repository CSV has 1,048,575 records and 21 columns. The file has no patient identifier and 812,049 records are exact duplicates across all columns. There are 38,133 unique rows in the 13 candidate clinical predictors. These repeated records limit independent evaluation; a record-level split may place exact copies in both partitions.

`PATIENT_TYPE=1` records carry unknown (97) ICU and intubation values. Known ICU and intubation labels occur in `PATIENT_TYPE=2`, so target construction on known labels implicitly restricts both tasks to hospitalized records. For the death task, `DATE_DIED` uses `9999-99-99` as no recorded death and all other non-missing values parse as dates under `%d/%m/%Y` in this source.

## Variables and cleaning

Feature groups: Demo (`AGE`, `SEX`); DemoComorb (Demo plus ten comorbidity flags); Clinical (DemoComorb plus `PNEUMONIA`). Targets and care-setting, pregnancy, service, and classification fields are excluded. This avoids direct target leakage and limits use of variables that are not part of the stated feature groups. Age codes 97/98/99, binary comorbidity codes 98/99, and pneumonia code 99 become missing. Median imputation and standardization apply to age; categorical missing values use most-frequent imputation and one-hot encoding. Estimators fit these steps using training data only.

The source data uses no patient ID. The runner groups identical rows in the complete 13-feature candidate matrix before stratified allocation. It uses this common split for the feature subsets, reducing exact-feature duplicate leakage while preserving fair row-level comparisons. It cannot guarantee that records from the same person never cross the split. Historical experiment files predate this safeguard and used stratified random row splits.

## Evaluation and methods

The experiment runner uses `random_state=42` and a nominal 80/20 held-out split. Metrics include accuracy, precision, recall, F1, ROC-AUC, average precision, and confusion-matrix counts. Average precision is the implementation stored under `pr_auc`. No threshold tuning on test outcomes is performed. SMOTEENN resamples only the training partition, with a default 100,000-row stratified training cap; the test partition is not resampled.

Raw preprocessing is the control. PCA retains 95% of training variance. LASSO means L1-penalized Logistic Regression with `SelectFromModel`, not a separate regression model. GUS is an explicitly approximate mutual-information `SelectKBest` proxy because the assigned paper does not provide enough detail for an exact reproduction. FPR uses scikit-learn `SelectFpr` with the ANOVA F-test. These are run independently.

## Historical result status

The historical output tables were generated using the previous row-random split. They serve as a record of executed experiments, not as a valid estimate of performance on unseen patients. Raw coverage is 45/45; PCA is 15/15 for Clinical; LASSO contains 15 Clinical rows at `C=1.0`; SMOTEENN contains 42/45 with three failures. GUS and FPR had no historical results when audited. A separate grouped split comparison completed 18 Clinical Logistic Regression runs (six methods x three targets); the per-method CSVs remain separate from historical outputs. The grouped grid includes a 100,000-row SMOTEENN training cap and is still a subset of the full model/feature grid.
