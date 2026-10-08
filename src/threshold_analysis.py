import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from preprocessing_clean import (
    load_data,
    prepare_target,
    split_data,
    build_preprocessor
)


def analyze_thresholds(target, df):

    print("\n" + "=" * 70)
    print(f"THRESHOLD ANALYSIS — {target}")
    print("=" * 70)

    # Prepare data
    X, y = prepare_target(df, target)

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Preprocessing
    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Logistic Regression baseline
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    # Get probabilities
    y_prob = model.predict_proba(
        X_test_processed
    )[:, 1]

    # Overall ranking metrics don't depend on threshold
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    # -----------------------------------------------------
    # Test different classification thresholds
    # -----------------------------------------------------

    thresholds = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80
    ]

    results = []

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    results_df = pd.DataFrame(results)

    print("\nThreshold comparison:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return results_df


if __name__ == "__main__":

    df = load_data()

    all_results = {}

    for target in [
        "ICU",
        "INTUBED",
        "DEATH"
    ]:

        results = analyze_thresholds(
            target,
            df
        )

        all_results[target] = results