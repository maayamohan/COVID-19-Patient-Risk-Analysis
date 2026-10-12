"""
Resumable experiment runner for COVID-19 Patient Risk Analysis.

Features:
- Saves each successful experiment to CSV immediately.
- Resumes by skipping experiments already present in the result CSV.
- Logs failures to a separate CSV.
- Keeps test data untouched by resampling.
- Caps SMOTEENN training rows by default to avoid prohibitively expensive
  neighbour cleaning on the full DEATH dataset. The cap applies only after
  the stratified train/test split and only to the training portion.
- Supports selecting targets, feature sets, and models from the CLI.

Examples:
    python src/run_experiments.py --engineering raw
    python src/run_experiments.py --engineering smoteenn --resume
    python src/run_experiments.py --engineering pca --targets ICU DEATH
    python src/run_experiments.py --engineering lasso --lasso-c 1.0
    python src/run_experiments.py --engineering smoteenn --max-train-samples 100000

IMPORTANT:
The SMOTEENN row cap changes the training sample size. Results produced with
a cap must be reported as capped-training experiments, not as full-data
SMOTEENN experiments. The held-out test set remains untouched.
"""

import os
import sys
import argparse
import traceback

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing_clean import prepare_target
from src.feature_engineering import get_feature_sets, build_feature_pipeline
from src.advanced_models import get_models


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_SMOTEENN_TRAIN_CAP = 100_000

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TARGETS = ["ICU", "INTUBED", "DEATH"]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def get_model_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise AttributeError(
        f"{type(model).__name__} provides neither predict_proba "
        "nor decision_function."
    )


# ============================================================
# RESULT / RESUME HELPERS
# ============================================================

KEY_COLUMNS = ["target", "feature_set", "engineering", "model", "lasso_C"]


def normalized_key(row):
    c_value = row.get("lasso_C")
    if pd.isna(c_value):
        c_value = None
    else:
        c_value = float(c_value)
    return (
        str(row.get("target")),
        str(row.get("feature_set")),
        str(row.get("engineering")),
        str(row.get("model")),
        c_value,
    )


def load_existing_results(results_file, resume):
    # Load existing rows even when the caller elects to rerun a configuration.
    # A non-resume run may replace matching keys, but must preserve unrelated
    # completed combinations already saved in this experiment file.
    if os.path.exists(results_file):
        try:
            existing = pd.read_csv(results_file)
            required = set(KEY_COLUMNS)
            if required.issubset(existing.columns):
                print(
                    f"Loaded {len(existing)} existing result row(s) "
                    f"from {results_file}"
                )
                return existing
            print(
                "Existing results file lacks required key columns; "
                "starting a new result table for this run."
            )
        except Exception as exc:
            print(f"Could not read existing results CSV: {exc}")
            print("Starting a new result table.")
    return pd.DataFrame()


def save_results(results_df, results_file):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)


def append_failure(failures_file, details):
    row = pd.DataFrame([details])
    write_header = not os.path.exists(failures_file)
    row.to_csv(
        failures_file,
        mode="a",
        header=write_header,
        index=False,
    )


# ============================================================
# ONE EXPERIMENT
# ============================================================

def run_single_experiment(
    train_df,
    test_df,
    target,
    feature_set_name,
    features,
    model_name,
    model,
    engineering_method,
    lasso_c,
    max_train_samples,
):
    print("\n" + "=" * 72)
    print(
        f"Target: {target} | Features: {feature_set_name} | "
        f"Engineering: {engineering_method} | Model: {model_name}"
    )
    print("=" * 72)

    # Use the predefined cohorts: Cohort 1 for training, Cohort 2 for testing.
    # Prepare targets independently because unknown outcomes may be filtered.
    X_train, y_train = prepare_target(train_df, target)
    X_test, y_test = prepare_target(test_df, target)

    candidate_features = get_feature_sets()["Clinical"]
    X_train = X_train[candidate_features].copy()
    X_test = X_test[candidate_features].copy()

    print(f"Training cohort samples: {len(X_train)}")
    print(f"Test cohort samples: {len(X_test)}")
    print(f"Training positive cases: {int(y_train.sum())}")
    print(f"Training positive rate: {y_train.mean():.4f}")
    print(f"Test positive cases: {int(y_test.sum())}")
    print(f"Test positive rate: {y_test.mean():.4f}")

    X_train = X_train[features].copy()
    X_test = X_test[features].copy()

    original_train_size = len(X_train)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # To keep SMOTEENN feasible on the very large DEATH target, take a
    # stratified subset from TRAIN only. The test set is never sampled.
    applied_train_cap = None
    if (
        engineering_method == "smoteenn"
        and max_train_samples is not None
        and len(X_train) > max_train_samples
    ):
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=max_train_samples,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
        applied_train_cap = max_train_samples
        print(
            f"Training subset: {len(X_train)} rows "
            f"(stratified from {original_train_size}); "
            "test set unchanged."
        )

    pipeline = build_feature_pipeline(
        features=features,
        engineering=engineering_method,
        lasso_C=lasso_c,
    )

    print("Fitting preprocessing / feature engineering...")

    if engineering_method == "smoteenn":
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours

        # Current feature_engineering.py names this step "preprocessing".
        if "preprocessing" not in pipeline.named_steps:
            raise KeyError(
                "Expected pipeline step 'preprocessing'. "
                f"Found: {list(pipeline.named_steps)}"
            )

        preprocessor = pipeline.named_steps["preprocessing"]
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        print(f"Before SMOTEENN: {X_train_processed.shape}")

        minority_count = min(y_train.value_counts())
        majority_count = max(y_train.value_counts())
        current_ratio = minority_count / majority_count

        smote = SMOTE(
            sampling_strategy=max(1 / 3, current_ratio + 0.01),
            random_state=RANDOM_STATE,
        )
        enn = EditedNearestNeighbours(
            sampling_strategy="all",
            n_neighbors=3,
        )

        X_train_processed, y_train = smote.fit_resample(
            X_train_processed, y_train
        )
        print(f"After SMOTE: {X_train_processed.shape}")

        X_train_processed, y_train = enn.fit_resample(
            X_train_processed, y_train
        )
        print(f"After SMOTEENN: {X_train_processed.shape}")

    else:
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        X_test_processed = pipeline.transform(X_test)

    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")
    print(f"Training {model_name}...")

    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    y_score = get_model_score(model, X_test_processed)

    metrics = calculate_metrics(y_test, y_pred, y_score)

    print("\nResults:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"{metric.upper():10s}: {metrics[metric]:.4f}")
    print("Confusion matrix:")
    print(
        confusion_matrix(y_test, y_pred, labels=[0, 1])
    )

    return {
        "target": target,
        "feature_set": feature_set_name,
        "engineering": engineering_method,
        "lasso_C": float(lasso_c) if engineering_method == "lasso" else None,
        "model": model_name,
        "n_input_features": len(features),
        "original_train_samples": original_train_size,
        "train_samples_before_resampling": len(X_train),
        "test_samples": len(X_test),
        "smoteenn_train_cap": applied_train_cap,
        "random_state": RANDOM_STATE,
        "split_method": "predefined_cohort_1_train_cohort_2_test",
        **metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run resumable COVID-19 patient-risk experiments."
    )
    parser.add_argument(
        "--engineering",
        choices=["raw", "smoteenn", "pca", "lasso", "gus", "fpr"],
        default="raw",
    )
    parser.add_argument("--lasso-c", type=float, default=1.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing results and skip matching completed experiments.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=DEFAULT_SMOTEENN_TRAIN_CAP,
        help=(
            "Maximum stratified training rows before SMOTEENN. "
            "Default: 100000. Use 0 to disable the cap."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGETS,
        default=TARGETS,
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["Demo", "DemoComorb", "Clinical"],
        default=None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["LogisticRegression", "RandomForest", "SVM", "MLP", "LightGBM"],
        default=None,
    )
    args = parser.parse_args()

    if args.lasso_c <= 0:
        parser.error("--lasso-c must be greater than zero.")
    if args.max_train_samples < 0:
        parser.error("--max-train-samples must be >= 0.")

    engineering_method = args.engineering
    max_train_samples = args.max_train_samples or None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Keep the historical row-random outputs untouched. New group-aware runs
    # use separate files so resume never treats an old split as equivalent.
    results_file = os.path.join(
        RESULTS_DIR, f"{engineering_method}_cohort_results.csv"
    )
    failures_file = os.path.join(
        RESULTS_DIR, f"{engineering_method}_cohort_failures.csv"
    )

    print("\n" + "=" * 72)
    print("COVID-19 PATIENT RISK ANALYSIS")
    print(f"ENGINEERING: {engineering_method.upper()}")
    if engineering_method == "lasso":
        print(f"LASSO C: {args.lasso_c}")
    if engineering_method == "smoteenn":
        print(
            "SMOTEENN training cap: "
            f"{max_train_samples if max_train_samples else 'disabled'}"
        )
    print("=" * 72)

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "dataset", "cohort_1.csv"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "dataset", "cohort_2.csv"))
    print(f"Cohort 1 (training) shape: {train_df.shape}")
    print(f"Cohort 2 (testing): shape {test_df.shape}")

    all_feature_sets = get_feature_sets()
    feature_sets = {
        name: feats
        for name, feats in all_feature_sets.items()
        if args.feature_sets is None or name in args.feature_sets
    }

    available_models = get_models()
    selected_models = [
        name for name in available_models
        if args.models is None or name in args.models
    ]

    targets = args.targets
    total = len(targets) * len(feature_sets) * len(selected_models)
    print(f"Targets: {targets}")
    print(f"Feature sets: {list(feature_sets)}")
    print(f"Models: {selected_models}")
    print(f"Experiments requested this run: {total}")

    results_df = load_existing_results(results_file, resume=True)
    completed = set()
    if not results_df.empty and set(KEY_COLUMNS).issubset(results_df.columns):
        for _, row in results_df.iterrows():
            completed.add(normalized_key(row))

    # Avoid duplicate entries if the CSV contains repeated keys.
    if not results_df.empty and set(KEY_COLUMNS).issubset(results_df.columns):
        results_df = results_df.drop_duplicates(subset=KEY_COLUMNS, keep="last")

    successes_this_run = 0
    skipped = 0
    failures = 0
    counter = 0

    for target in targets:
        for feature_set_name, features in feature_sets.items():
            for model_name in selected_models:
                counter += 1
                lasso_key = float(args.lasso_c) if engineering_method == "lasso" else None
                key = (target, feature_set_name, engineering_method, model_name, lasso_key)

                if args.resume and key in completed:
                    skipped += 1
                    print(
                        f"\n[{counter}/{total}] SKIP already completed: "
                        f"{target} / {feature_set_name} / {model_name}"
                    )
                    continue

                print(f"\nExperiment {counter}/{total}")
                model = get_models()[model_name]

                try:
                    result = run_single_experiment(
                        train_df=train_df,
                        test_df=test_df,
                        target=target,
                        feature_set_name=feature_set_name,
                        features=features,
                        model_name=model_name,
                        model=model,
                        engineering_method=engineering_method,
                        lasso_c=args.lasso_c,
                        max_train_samples=max_train_samples,
                    )

                    new_row = pd.DataFrame([result])
                    if results_df.empty:
                        results_df = new_row
                    else:
                        results_df = pd.concat(
                            [results_df, new_row], ignore_index=True
                        )

                    results_df = results_df.drop_duplicates(
                        subset=KEY_COLUMNS, keep="last"
                    )
                    save_results(results_df, results_file)
                    completed.add(key)
                    successes_this_run += 1
                    print(f"Saved progress immediately to: {results_file}")

                except KeyboardInterrupt:
                    print(
                        "\nInterrupted by user. All previously completed "
                        "experiments remain saved in the results CSV."
                    )
                    save_results(results_df, results_file)
                    raise

                except Exception as exc:
                    failures += 1
                    print(
                        f"\nERROR: {target} / {feature_set_name} / "
                        f"{model_name}: {exc!r}"
                    )
                    append_failure(
                        failures_file,
                        {
                            "target": target,
                            "feature_set": feature_set_name,
                            "engineering": engineering_method,
                            "model": model_name,
                            "lasso_C": (
                                args.lasso_c
                                if engineering_method == "lasso"
                                else None
                            ),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(f"Failure details appended to: {failures_file}")

    print("\n" + "=" * 72)
    print("EXPERIMENT RUN SUMMARY")
    print("=" * 72)
    print(f"Successful this run: {successes_this_run}")
    print(f"Skipped as already completed: {skipped}")
    print(f"Failures this run: {failures}")
    print(f"Results currently saved: {len(results_df)}")
    print(f"Results CSV: {results_file}")
    if os.path.exists(failures_file):
        print(f"Failure log: {failures_file}")

    if not results_df.empty:
        subset = results_df[
            results_df["target"].isin(targets)
            & results_df["feature_set"].isin(feature_sets.keys())
            & results_df["model"].isin(selected_models)
            & (results_df["engineering"] == engineering_method)
        ]
        if not subset.empty:
            print("\nTop results by ROC-AUC in the selected result subset:")
            columns = [
                "target", "feature_set", "engineering",
                "model", "roc_auc", "pr_auc", "f1"
            ]
            print(
                subset[columns]
                .sort_values("roc_auc", ascending=False)
                .head(10)
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()
