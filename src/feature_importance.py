import pandas as pd

from sklearn.linear_model import LogisticRegression

from preprocessing_clean import (
    load_data,
    prepare_target,
    split_data,
    build_preprocessor
)


# ---------------------------------------------------------
# Extract and display Logistic Regression coefficients
# ---------------------------------------------------------

def analyze_features(target, df):

    print("\n" + "=" * 70)
    print(f"FEATURE IMPORTANCE — {target}")
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

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Get the actual feature names after one-hot encoding
    feature_names = preprocessor.get_feature_names_out()

    # Logistic Regression
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_processed,
        y_train
    )

    # Get coefficients
    coefficients = model.coef_[0]

    # Create dataframe
    feature_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients
    })

    # Absolute coefficient = strength of association
    feature_df["absolute_coefficient"] = (
        feature_df["coefficient"].abs()
    )

    # Sort by absolute importance
    feature_df = feature_df.sort_values(
        "absolute_coefficient",
        ascending=False
    )

    print("\nTop 10 features by absolute coefficient:")
    print(
        feature_df[
            [
                "feature",
                "coefficient"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print("\nTop positive coefficients:")
    print(
        feature_df.sort_values(
            "coefficient",
            ascending=False
        )[
            [
                "feature",
                "coefficient"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print("\nTop negative coefficients:")
    print(
        feature_df.sort_values(
            "coefficient",
            ascending=True
        )[
            [
                "feature",
                "coefficient"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    return feature_df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    all_results = {}

    for target in [
        "ICU",
        "INTUBED",
        "DEATH"
    ]:

        results = analyze_features(
            target,
            df
        )

        all_results[target] = results