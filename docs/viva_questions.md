# Viva questions and short answers

### What problem does the project study?
It tests whether demographic and routinely recorded clinical variables can rank COVID-19 records by severe outcomes: ICU admission, intubation, and recorded death. These are retrospective prediction experiments, not causal analysis or clinical validation.

### How does this relate to the assigned paper?
Zhan and Li's report uses CT radiomics, clinical features, lab results, and radiologist findings. The original patient-level data were not available for this project, so the approved public Mexican surveillance CSV was substituted. This changes both the inputs and the study design; we do not claim exact replication.

### What is the dataset size and what is a major quality concern?
The included CSV has 1,048,575 rows and 21 columns. It has no patient identifier and 812,049 exact duplicate rows. This makes ordinary row-random test scores vulnerable to leakage and prevents a claim of patient-level independence.

### How are the labels built?
ICU and INTUBED use only known codes 1 and 2, mapped to positive and negative. Other codes are unknown and excluded. For DEATH, `9999-99-99` means no recorded death and a valid `%d/%m/%Y` date means death. Malformed dates are excluded. Known ICU and intubation labels occur only for `PATIENT_TYPE=2`, so those tasks cover hospitalized records.

### How imbalanced are the targets?
ICU prevalence is 8.75% among 192,543 known records. Intubation prevalence is 17.46% among 192,706 known records. Recorded death prevalence is 7.34% of 1,048,575 rows. A majority-class model could therefore have high accuracy while missing many positives.

### Which features are used, and what is excluded?
Demo uses age and sex. DemoComorb adds ten comorbidity flags. Clinical adds pneumonia. Targets, `PATIENT_TYPE`, `DATE_DIED`, `INTUBED`, `ICU`, pregnancy, classification, medical unit, and USMER are excluded to reduce direct leakage and proxy use.

### What do the sentinel values mean?
Age values 97/98/99, comorbidity values 98/99, and pneumonia 99 encode unknown or unavailable values. They become missing and are imputed within a training-fitted pipeline. They are not treated as ordinary measurements.

### Why stratify the split, and what changed for duplicates?
Stratification aims to preserve the outcome proportion. The maintained split also groups identical predictor rows so identical feature vectors do not occur in both partitions. There is no patient ID, so this does not prove distinct-patient separation. All pre-existing experiment CSVs used ordinary row-random splits and remain exploratory.

### Why fit preprocessing only on training data?
An imputer, scaler, PCA, or selector learns quantities from data. Fitting it on test data lets test distribution information influence the model. The pipeline fits those transformations on training rows and only calls `transform` on held-out data.

### Why these models?
Logistic Regression is a linear baseline; Random Forest captures nonlinear splits; Linear SVM offers a margin-based linear classifier at this dataset size; MLP models nonlinear interactions; LightGBM is a boosted tree model. Comparing them reveals different ranking and threshold behavior.

### Explain the metrics.
Precision is the fraction of predicted positives that are positive. Recall is the fraction of positives found. F1 balances precision and recall. ROC-AUC evaluates ranking across thresholds. Average precision summarizes the precision-recall curve and is stored as `pr_auc`. Confusion counts show the actual error trade-off.

### Why can ROC-AUC look reasonable while F1 or precision is poor?
ROC-AUC evaluates ordering across thresholds; default classification uses one threshold. Under class imbalance, many false positives can lower precision at that threshold even while positive cases tend to receive higher scores overall.

### What is SMOTEENN and what happened here?
SMOTE synthesizes minority-class examples and ENN removes nearby noisy examples. In the historical run, 100,000 training rows were sampled after splitting; the test set was untouched. ENN removed a class for three INTUBED/Demo model combinations, so those experiments failed and are logged. Capped training also makes SMOTEENN results a different training regime.

### What are PCA, LASSO, GUS, and FPR here?
PCA keeps components explaining about 95% of training variance. LASSO is L1 Logistic Regression used as a feature selector. GUS is approximated with mutual-information `SelectKBest` because the paper does not specify enough implementation details. FPR uses scikit-learn `SelectFpr` with an ANOVA F-test. They are separate experiments.

### Was the classification threshold tuned?
The saved runs use each estimator's default predictions. The held-out test set was not used to choose a threshold. A future threshold study should select it using training-only cross-validation or a validation split.

### What are the best saved results?
Among historical saved rows by F1: ICU Clinical raw LightGBM has F1 0.2200 (precision 0.1275, recall 0.8028, ROC-AUC 0.6761, average precision 0.1621). Intubation Clinical raw LightGBM has F1 0.3526 (precision 0.2383, recall 0.6778, ROC-AUC 0.6428, AP 0.2499). Death Clinical SMOTEENN LightGBM has F1 0.5090 (precision 0.3959, recall 0.7126, ROC-AUC 0.9148, AP 0.4587). They use historical random splits and must not be presented as leakage-controlled performance.

### Did feature engineering help consistently?
No single method dominates all targets and metrics. The saved files differ in coverage, SMOTEENN caps its training rows, and historical splits permit exact duplicates across partitions. These results are exploratory and need a complete grouped-split grid before a fair method ranking.

### Did adding comorbidities or pneumonia help?
In the historical raw grid, mean F1 across five models was higher for Clinical than Demo: ICU 0.171 vs 0.135, intubation 0.274 vs 0.255, and death 0.455 vs 0.259. The result is descriptive because those random splits could place exact duplicate rows on both sides. A matched grouped-split comparison across feature sets remains future work.

### Why are some experiments missing or failed?
Saved coverage is raw 45/45, PCA 15 Clinical rows, LASSO 13 Clinical rows, SMOTEENN 42/45, and no GUS/FPR result files at audit time. SMOTEENN's three failures came from ENN removing one class. Results must be described with these coverage gaps.

### What are the main limitations and ethical concerns?
There are repeated rows, no patient identifier, class imbalance, limited features, unclear predictor timing, and no external validation. Predictions could reflect data collection patterns. These models should not guide care, triage, or resource allocation.

### How should you describe your contribution?
Describe only work you personally performed. The repository separates preprocessing, feature engineering, model definitions, experiment orchestration, saved results, and documentation. Do not infer team-member names or individual contributions from missing repository metadata.
