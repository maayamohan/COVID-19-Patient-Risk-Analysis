# Experiment results audit

## Provenance

The checked-in working tree already contained uncommitted historical result files. I preserved them and audited their rows rather than treating the prompt's example metrics as ground truth. These files came from a stratified random row split (`random_state=42`), not a patient-level split. The source contains 812,049 exact duplicate rows and has no patient ID, so a duplicate can occur in both partitions. Metrics below are executed results but are potentially optimistic and are not claims about unseen-patient performance.

## Coverage

| Method | Saved rows | Coverage / notes |
|---|---:|---|
| Raw | 45 | 3 targets x 3 feature groups x 5 models |
| PCA | 15 | Clinical x 3 targets x 5 models |
| LASSO | 15 | Clinical, C=1.0; 5 models for each target |
| SMOTEENN | 42 | 45 requested, 100,000-row training cap, test set unchanged |
| GUS-inspired MI selector | 0 historical; 3 grouped | Clinical Logistic Regression, one per target |
| FPR / ANOVA F-test | 0 historical; 3 grouped | Clinical Logistic Regression, one per target |

Three SMOTEENN combinations failed: INTUBED / Demo / LogisticRegression, RandomForest, and SVM. The ENN step removed one class from the sampled training data. Logistic Regression and SVM then refused to fit; the Random Forest fit but exposed a one-class probability output when scoring. These rows remain in `results/smoteenn_failures.csv`; they were not silently replaced.

## Highest observed F1 by target

Selection rule: highest F1 among the saved rows for each target, across methods. This is a descriptive choice for oral discussion, not a clinically optimized threshold or a fair grouped-split comparison. The saved test threshold is each estimator's default class prediction threshold.

| Target | Method / group / model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|---:|---:|
| ICU | Raw / Clinical / LightGBM | 0.5016 | 0.1275 | 0.8028 | 0.2200 | 0.6761 | 0.1621 |
| INTUBED | Raw / Clinical / LightGBM | 0.5654 | 0.2383 | 0.6778 | 0.3526 | 0.6428 | 0.2499 |
| DEATH | SMOTEENN / Clinical / LightGBM | 0.8991 | 0.3959 | 0.7126 | 0.5090 | 0.9148 | 0.4587 |

The class imbalance matters: precision is low for ICU and intubation despite high recall. SMOTEENN has the largest observed death F1, but its capped training sample and historical split make it incomparable to a patient-independent clinical claim. The prompt's cited baseline values are close to, but not exactly, the stored raw Clinical Logistic Regression rows; use the figures above from the CSV.

Within the historical raw grid, mean metrics across the five models rise from Demo to Clinical. Mean F1 is ICU 0.135 to 0.171, intubation 0.255 to 0.274, and death 0.259 to 0.455. This pattern is consistent with additional comorbidity and pneumonia information improving these particular saved runs, but exact duplicates can inflate the estimates and the feature-group comparison has not yet been repeated using identical group-aware splits.

## Target prevalence in the supplied data

ICU includes 192,543 known labels, of which 16,858 are positive (8.75%). Intubation includes 192,706 known labels, of which 33,656 are positive (17.46%). Recorded death includes 1,048,575 rows, of which 76,942 have a valid death date (7.34%). The ICU/intubation cohorts are hospitalized cases because all known labels occur at `PATIENT_TYPE=2`.

## Corrected runs

Future invocations of `src/run_experiments.py` use a stratified grouped split that keeps identical predictor rows together and write to separate `*_grouped_results.csv` files. The old files remain unchanged so their provenance is clear. This grouping still does not identify distinct patients; the source has no patient identifier.

The corrected grouped experiment grid has 18 runs: one Clinical Logistic Regression per target and each of raw, PCA, LASSO, GUS, FPR, and SMOTEENN. It uses the same grouped split for comparable feature methods, except that SMOTEENN caps the training portion at 100,000 rows. The held-out set remains unchanged.

| Target | Raw F1 | PCA F1 | LASSO F1 | GUS F1 | FPR F1 | SMOTEENN F1 |
|---|---:|---:|---:|---:|---:|---:|
| ICU | 0.2108 | 0.2103 | 0.2099 | 0.2097 | 0.2107 | 0.1823 |
| INTUBED | 0.3544 | 0.3543 | 0.3512 | 0.3529 | 0.3548 | 0.3008 |
| DEATH | 0.4781 | 0.4768 | 0.4819 | 0.4684 | 0.4776 | 0.4886 |

Complete metrics, sample counts, and protocol markers are saved in the matching `*_grouped_results.csv` files. Across the one-model grouped subset, the methods produce similar default-threshold F1 for ICU and intubation. Capped SMOTEENN has the highest death F1 in this subset, while all methods remain subject to threshold and dataset limitations. These 18 runs are not the full five-model, three-feature-group experiment grid.
