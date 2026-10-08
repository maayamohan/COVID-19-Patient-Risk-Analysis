"""
Paper-inspired feature engineering for the COVID-19 patient risk project.

Feature-engineering variants:
1. Raw
2. SMOTEENN
3. PCA
4. LASSO
5. GUS
6. FPR / F-classif

Important:
All transformations are designed to be placed inside an
imblearn Pipeline so that fitting happens only on training data.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SelectFpr,
    f_classif,
    mutual_info_classif
)

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# FEATURE SETS
# ============================================================

DEMO_FEATURES = [
    "AGE",
    "SEX"
]

COMORBIDITY_FEATURES = [
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMSUPR",
    "HIPERTENSION",
    "OTHER_DISEASE",
    "CARDIOVASCULAR",
    "OBESITY",
    "RENAL_CHRONIC",
    "TOBACCO"
]

CLINICAL_FEATURES = [
    *DEMO_FEATURES,
    *COMORBIDITY_FEATURES,
    "PNEUMONIA"
]


def get_feature_sets():
    """
    Return the feature sets used in the paper-inspired experiments.

    Demo:
        AGE + SEX

    DemoComorb:
        AGE + SEX + comorbidities

    Clinical:
        DemoComorb + PNEUMONIA
    """

    return {
        "Demo": DEMO_FEATURES,
        "DemoComorb": [
            *DEMO_FEATURES,
            *COMORBIDITY_FEATURES
        ],
        "Clinical": CLINICAL_FEATURES
    }


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(features):
    """
    Build leakage-safe preprocessing.

    AGE:
        median imputation + standardization

    Categorical / binary variables:
        most-frequent imputation + one-hot encoding
    """

    numerical_features = [
        feature for feature in features
        if feature == "AGE"
    ]

    categorical_features = [
        feature for feature in features
        if feature != "AGE"
    ]

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="if_binary"
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# SMOTEENN
# ============================================================

def build_smoteenn():
    """
    Paper-inspired SMOTEENN.

    We use a 1:3 sampling strategy rather than forcing a full
    1:1 balance because the dataset is very large and highly
    imbalanced, especially for ICU/MV/death.

    This sampler is applied only to training data when used
    inside an imblearn Pipeline.
    """

    smote = SMOTE(
        sampling_strategy=1 / 3,
        random_state=42
    )

    enn = EditedNearestNeighbours(
        sampling_strategy="all",
        n_neighbors=3
    )

    return SMOTEENN(
        smote=smote,
        enn=enn
    )


# ============================================================
# LASSO
# ============================================================

def build_lasso_selector(C=1.0):
    """
    LASSO-style feature selection using L1-regularized
    logistic regression.

    The paper explores different C values, so our experiments
    can test:

        C = 0.2
        C = 0.5
        C = 1
        C = 30
        C = 50
    """

    lasso_model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    selector = SelectFromModel(
        estimator=lasso_model,
        threshold="mean"
    )

    return selector


# ============================================================
# PCA
# ============================================================

def build_pca():
    """
    PCA retaining 95% of the variance.

    PCA is fitted only on the training data when used inside
    the pipeline.
    """

    return PCA(
        n_components=0.95,
        random_state=42
    )


# ============================================================
# GUS
# ============================================================

def build_gus():
    """
    Generic Univariate Selection proxy.

    The paper mentions GUS but does not provide enough detail
    in the paper to reproduce its exact implementation.

    Therefore we use mutual information as a defensible
    generic univariate feature-selection approach.

    This must be described as a paper-inspired approximation,
    not an exact reproduction.
    """

    return SelectKBest(
        score_func=mutual_info_classif,
        k="all"
    )


# ============================================================
# FPR / F-CLASSIF
# ============================================================

def build_fpr():
    """
    False Positive Rate feature selection using ANOVA
    F-classification criterion.

    This follows the paper's description of the FPR route
    using F-classif.
    """

    return SelectFpr(
        score_func=f_classif,
        alpha=0.05
    )


# ============================================================
# ENGINEERING COMPONENT
# ============================================================

def get_engineering_step(method):
    """
    Return the feature-engineering component for a given method.

    Supported:
        raw
        smoteenn
        pca
        lasso
        gus
        fpr
    """

    method = method.lower()

    if method == "raw":
        return None

    if method == "smoteenn":
        return build_smoteenn()

    if method == "pca":
        return build_pca()

    if method == "lasso":
        return build_lasso_selector(C=1.0)

    if method == "gus":
        return build_gus()

    if method == "fpr":
        return build_fpr()

    raise ValueError(
        f"Unknown feature engineering method: {method}"
    )


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_feature_pipeline(
    features,
    engineering="raw",
    lasso_C=1.0
):
    """
    Build a complete preprocessing + feature-engineering pipeline.

    NOTE:
    For SMOTEENN, the sampler is inserted after preprocessing.

    For PCA/LASSO/GUS/FPR, the selector/transformation is inserted
    after preprocessing.
    """

    preprocessor = build_preprocessor(features)

    engineering = engineering.lower()

    steps = [
        ("preprocessing", preprocessor)
    ]

    if engineering == "raw":
        pass

    elif engineering == "smoteenn":
        steps.append(
            ("smoteenn", build_smoteenn())
        )

    elif engineering == "pca":
        steps.append(
            ("pca", build_pca())
        )

    elif engineering == "lasso":
        steps.append(
            (
                "lasso",
                build_lasso_selector(C=lasso_C)
            )
        )

    elif engineering == "gus":
        steps.append(
            ("gus", build_gus())
        )

    elif engineering == "fpr":
        steps.append(
            ("fpr", build_fpr())
        )

    else:
        raise ValueError(
            f"Unknown engineering method: {engineering}"
        )

    return ImbPipeline(steps=steps)


# ============================================================
# AVAILABLE EXPERIMENTS
# ============================================================

def get_engineering_methods():
    """
    Return all feature-engineering variants.
    """

    return [
        "raw",
        "smoteenn",
        "pca",
        "lasso",
        "gus",
        "fpr"
    ]


def get_lasso_C_values():
    """
    C values inspired by the paper's LASSO experiments.
    """

    return [
        0.2,
        0.5,
        1.0,
        30.0,
        50.0
    ]