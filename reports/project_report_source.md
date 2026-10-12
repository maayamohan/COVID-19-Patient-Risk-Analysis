# COVID-19 Patient Risk Analysis
### A tabular adaptation of a CT-based triage research question

**UE24CS352A Machine Learning mini-project**

## Abstract

This project evaluates whether demographic and recorded clinical variables can rank COVID-19 cases by ICU admission, intubation, and recorded death. It adapts the outcomes studied by Zhan and Li using the public Mexican surveillance CSV included in the repository. It does not reproduce their CT/radiomics methodology. The source contains 1,048,575 rows, 21 columns, and 812,049 exact duplicate rows, with no patient identifier. Consequently, historical random-row test metrics can be optimistic and do not establish patient-level generalization.

## Data and method

ICU and intubation targets use only known codes 1 (positive) and 2 (negative); unknown codes are excluded. All known labels for these tasks occur among hospitalized cases (`PATIENT_TYPE=2`). Death is 1 for a valid `%d/%m/%Y` date and 0 for `9999-99-99`; malformed dates are excluded. Candidate groups are Demo (age, sex), DemoComorb (plus ten comorbidities), and Clinical (plus pneumonia). Target columns, `PATIENT_TYPE`, pregnancy, classification, and service/unit variables are excluded.

The known ICU cohort has 192,543 rows with 16,858 positives (8.75%). Intubation has 192,706 known rows with 33,656 positives (17.46%). Recorded death has 76,942 positive records among 1,048,575 rows (7.34%).

Age sentinels 97/98/99, comorbidity sentinels 98/99, and pneumonia 99 are treated as missing. Training-fitted median/mode imputation, age scaling, and categorical one-hot encoding occur inside the model pipeline. New splits use stratified grouping on identical predictor rows; this prevents exact feature-vector copies from crossing partitions, but cannot establish patient identity. Models are Logistic Regression, Random Forest, Linear SVM, MLP, and LightGBM. Independent engineering modes include raw, PCA (95% variance), L1 Logistic Regression selection, capped SMOTEENN, mutual-information selection (GUS-inspired proxy), and ANOVA F-test FPR. SMOTEENN affects training only and caps training at 100,000 rows by default.

## Results

Historical results use stratified random row splits, so they remain exploratory. The table selects the highest saved F1 per outcome; class predictions use estimator defaults. Average precision is reported as PR-AUC in the original CSVs.

| Outcome | Highest saved F1 configuration | Precision | Recall | F1 | ROC-AUC | AP |
|---|---|---:|---:|---:|---:|---:|
| ICU | Raw Clinical LightGBM | 0.1275 | 0.8028 | 0.2200 | 0.6761 | 0.1621 |
| Intubation | Raw Clinical LightGBM | 0.2383 | 0.6778 | 0.3526 | 0.6428 | 0.2499 |
| Death | SMOTEENN Clinical LightGBM | 0.3959 | 0.7126 | 0.5090 | 0.9148 | 0.4587 |

The historical grid has 45 raw rows, 15 Clinical PCA rows, 15 Clinical LASSO rows at C=1.0, and 42 SMOTEENN rows. Three INTUBED/Demo SMOTEENN fits failed after ENN removed one class. Historical GUS and FPR results were absent. A new matched subset completed all six methods across three targets using Clinical features and Logistic Regression (18 runs, no failures). The SMOTEENN group run used a stratified 100,000-row training cap.

In the matched grouped subset, F1 across Raw/PCA/LASSO/GUS/FPR/SMOTEENN is ICU 0.2108/0.2103/0.2099/0.2097/0.2107/0.1823; intubation 0.3544/0.3543/0.3512/0.3529/0.3548/0.3008; and death 0.4781/0.4768/0.4819/0.4684/0.4776/0.4886. This comparison fixes the model and feature group and keeps identical predictor rows grouped. SMOTEENN uses a training cap, so its training regime differs.

## Conclusion and limitations

The saved runs show useful ranking for death and weaker positive-class precision for ICU and intubation. The results do not establish clinical utility. Duplicate rows, the lack of patient IDs, imbalanced outcomes, uncertain predictor timing, and the dataset substitution all limit interpretation. A complete grouped-split experiment grid and external validation would be needed before stronger claims. This project is for coursework and must not guide patient care.

## References

Zhan, X. and Li, Y. “CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients.” CS229 Final Report (assigned research reference).

UE24CS352A, *Mini-Project Assignment Guidelines*, supplied course handout.
