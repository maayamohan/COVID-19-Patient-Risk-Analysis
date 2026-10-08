import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "dataset/Covid Data.csv"

FEATURES = [
    "SEX",
    "AGE",
    "PNEUMONIA",
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMSUPR",
    "HIPERTENSION",
    "OTHER_DISEASE",
    "CARDIOVASCULAR",
    "OBESITY",
    "RENAL_CHRONIC",
    "TOBACCO",
]


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_data():
    """Load the COVID-19 dataset."""

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {df.shape}")

    return df


# ---------------------------------------------------------
# Prepare target
# ---------------------------------------------------------

def prepare_target(df, target):
    """
    Prepare features X and binary target y.

    Supported targets:
        - ICU
        - INTUBED
        - DEATH
    """

    df = df.copy()

    # -------------------------
    # ICU
    # -------------------------
    if target == "ICU":

        # Keep only known ICU outcomes
        df = df[df["ICU"].isin([1, 2])].copy()

        # 1 = ICU admission
        # 2 = No ICU admission
        y = df["ICU"].map({
            1: 1,
            2: 0
        })

    # -------------------------
    # Mechanical ventilation
    # -------------------------
    elif target == "INTUBED":

        # Keep only known intubation outcomes
        df = df[df["INTUBED"].isin([1, 2])].copy()

        # 1 = intubated
        # 2 = not intubated
        y = df["INTUBED"].map({
            1: 1,
            2: 0
        })

    # -------------------------
    # Death
    # -------------------------
    elif target == "DEATH":

        # 9999-99-99 = no recorded death
        y = (
            df["DATE_DIED"]
            .astype(str)
            .ne("9999-99-99")
            .astype(int)
        )

    else:
        raise ValueError(
            "target must be one of: ICU, INTUBED, DEATH"
        )

    # Select only our predictor variables
    X = df[FEATURES].copy()

    # -----------------------------------------------------
    # Convert dataset sentinel values into missing values
    # -----------------------------------------------------

    # 98/99 represent unknown/missing values
    X = X.replace([97, 98, 99], np.nan)

    y.name = target
    return X, y


# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------

def split_data(X, y, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessor():

    categorical_features = [
        "SEX",
        "PNEUMONIA",
        "DIABETES",
        "COPD",
        "ASTHMA",
        "INMSUPR",
        "HIPERTENSION",
        "OTHER_DISEASE",
        "CARDIOVASCULAR",
        "OBESITY",
        "RENAL_CHRONIC",
        "TOBACCO",
    ]

    numerical_features = [
        "AGE"
    ]

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# Quick test
# ---------------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    for target in ["ICU", "INTUBED", "DEATH"]:

        print("\n" + "=" * 60)
        print(f"TARGET: {target}")
        print("=" * 60)

        X, y = prepare_target(df, target)

        print("X shape:", X.shape)
        print("y shape:", y.shape)

        print("\nTarget distribution:")
        print(y.value_counts())

        print("\nTarget proportions:")
        print(y.value_counts(normalize=True))

        X_train, X_test, y_train, y_test = split_data(
            X,
            y
        )

        print("\nTrain shape:", X_train.shape)
        print("Test shape :", X_test.shape)

        preprocessor = build_preprocessor()

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        print(
            "Processed train shape:",
            X_train_processed.shape
        )

        print(
            "Processed test shape :",
            X_test_processed.shape
        )