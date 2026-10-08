"""
Model definitions for the paper-inspired COVID-19 risk prediction pipeline.

Models:
1. Logistic Regression
2. Random Forest
3. SVM
4. MLP
5. LightGBM
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from lightgbm import LGBMClassifier


def get_models():
    """
    Return the models used in the experiments.

    All models use class weighting or an equivalent strategy
    where supported because ICU, mechanical ventilation and
    mortality outcomes are imbalanced.
    """

    models = {

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------
        # LinearSVC is used instead of kernel SVC because our
        # dataset is very large. Kernel SVC would be
        # computationally impractical on hundreds of thousands
        # of samples.
        "SVM": LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ),

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        ),

        # ----------------------------------------------------
        # LightGBM
        # ----------------------------------------------------
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    }

    return models


def get_model_names():
    """
    Return model names.
    """

    return [
        "LogisticRegression",
        "RandomForest",
        "SVM",
        "MLP",
        "LightGBM"
    ]