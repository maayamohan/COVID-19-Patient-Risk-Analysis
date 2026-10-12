import numpy as np
import pandas as pd

from src.preprocessing_clean import FEATURES, prepare_target, split_data
from src.feature_engineering import get_feature_sets, build_feature_pipeline
from src.run_experiments import (
    calculate_metrics, normalized_key, load_existing_results, KEY_COLUMNS,
)


def make_rows(n=20):
    d = {name: np.ones(n, dtype=int) for name in FEATURES}
    d["AGE"] = np.arange(20, 20 + n)
    d["SEX"] = np.tile([1, 2], n // 2)
    d["ICU"] = np.tile([1, 2], n // 2)
    d["INTUBED"] = np.tile([2, 1], n // 2)
    d["DATE_DIED"] = np.tile(["9999-99-99", "01/02/2021"], n // 2)
    return pd.DataFrame(d)


def test_target_mapping_and_unknown_targets():
    d = make_rows()
    d.loc[0, "ICU"] = 97
    X, y = prepare_target(d, "ICU")
    assert len(y) == 19
    assert set(y.unique()) == {0, 1}
    assert X.columns.tolist() == FEATURES
    _, death = prepare_target(d, "DEATH")
    assert death.tolist() == [0, 1] * 10


def test_sentinels_become_missing_without_corrupting_sex():
    d = make_rows()
    d.loc[0, "AGE"] = 97
    d.loc[1, "PNEUMONIA"] = 99
    d.loc[2, "DIABETES"] = 98
    X, _ = prepare_target(d, "ICU")
    assert pd.isna(X.loc[0, "AGE"])
    assert pd.isna(X.loc[1, "PNEUMONIA"])
    assert pd.isna(X.loc[2, "DIABETES"])
    assert X["SEX"].isin([1, 2]).all()


def test_death_excludes_invalid_dates_and_maps_valid_date():
    d = make_rows()
    d.loc[0, "DATE_DIED"] = "not-a-date"
    X, y = prepare_target(d, "DEATH")
    assert len(X) == 19
    assert y.isin([0, 1]).all()


def test_feature_groups_exclude_outcome_and_disposition_fields():
    sets = get_feature_sets()
    assert sets["Demo"] == ["AGE", "SEX"]
    assert set(sets["DemoComorb"]).issubset(sets["Clinical"])
    assert "PNEUMONIA" in sets["Clinical"]
    excluded = {"ICU", "INTUBED", "DATE_DIED", "PATIENT_TYPE", "CLASIFFICATION_FINAL"}
    assert not excluded.intersection(set().union(*map(set, sets.values())))


def test_group_split_keeps_identical_predictor_rows_together():
    ages = np.repeat(np.arange(20, 30), 2)
    X = pd.DataFrame({"AGE": ages, "SEX": [1] * len(ages)})
    y = pd.Series([int(age >= 25) for age in ages])
    Xtr, Xte, ytr, yte = split_data(X, y, test_size=0.25, random_state=42)
    assert set(map(tuple, Xtr.to_numpy())).isdisjoint(set(map(tuple, Xte.to_numpy())))
    assert set(ytr) == {0, 1} and set(yte) == {0, 1}


def test_pipeline_construction_and_training_only_transform_fit():
    X = pd.DataFrame({"AGE": [20, 30, 40, 50, 60, 70], "SEX": [1, 2, 1, 2, 1, 2]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    pipeline = build_feature_pipeline(["AGE", "SEX"], "raw")
    transformed = pipeline.fit_transform(X.iloc[:4], y.iloc[:4])
    held_out = pipeline.transform(X.iloc[4:])
    assert transformed.shape[1] == held_out.shape[1]
    assert "preprocessing" in pipeline.named_steps


def test_metric_aggregation_and_resume_key():
    m = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.4, 0.8, 0.9])
    assert m["true_negative"] == 1 and m["true_positive"] == 2
    assert 0 <= m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1
    assert normalized_key({"target": "ICU", "feature_set": "Clinical", "engineering": "lasso", "model": "SVM", "lasso_C": 1})[-1] == 1.0


def test_resume_reads_results_and_duplicate_keys_can_be_collapsed(tmp_path):
    path = tmp_path / "previous.csv"
    row = {
        "target": "DEATH", "feature_set": "Clinical", "engineering": "raw",
        "model": "LogisticRegression", "lasso_C": np.nan, "f1": 0.4,
    }
    pd.DataFrame([row, row]).to_csv(path, index=False)
    loaded = load_existing_results(path, resume=True)
    assert len(loaded) == 2
    assert normalized_key(loaded.iloc[0]) == normalized_key(loaded.iloc[1])
    assert len(loaded.drop_duplicates(subset=KEY_COLUMNS, keep="last")) == 1

