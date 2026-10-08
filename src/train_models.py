import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from preprocessing_clean import (
    load_data,
    prepare_target,
    split_data,
    build_preprocessor
)


# ---------------------------------------------------------
# Train and evaluate Logistic Regression
# ---------------------------------------------------------

def train_logistic_regression(target, df):

    print("\n" + "=" * 70)
    print(f"LOGISTIC REGRESSION — {target}")
    print("=" * 70)

    # Prepare target-specific dataset
    X, y = prepare_target(df, target)

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessor()

    # Fit preprocessing ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using the training preprocessing
    X_test_processed = preprocessor.transform(X_test)

    print("\nTraining data:", X_train_processed.shape)
    print("Test data:", X_test_processed.shape)

    # -----------------------------------------------------
    # Logistic Regression
    # -----------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    # Train
    model.fit(
        X_train_processed,
        y_train
    )

    # Predictions
    y_pred = model.predict(
        X_test_processed
    )

    # Probability of positive class
    y_prob = model.predict_proba(
        X_test_processed
    )[:, 1]

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # ROC-AUC
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # PR-AUC
    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    return {
        "target": target,
        "model": "Logistic Regression",
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    results = []

    for target in [
        "ICU",
        "INTUBED",
        "DEATH"
    ]:

        result = train_logistic_regression(
            target,
            df
        )

        results.append(result)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )