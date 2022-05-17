import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from evalml.pipelines import UnsupervisedPipeline


def test_unsupervised_init():
    pipeline = UnsupervisedPipeline(
        component_graph=["Imputer", "One Hot Encoder", "DBSCAN Clusterer"]
    )
    assert pipeline.parameters == {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
        },
        "One Hot Encoder": {
            "top_n": 10,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "DBSCAN Clusterer": {"eps": 0.5, "min_samples": 5, "leaf_size": 30, "n_jobs": -1},
    }
    assert pipeline.name == "DBSCAN Clusterer w/ Imputer + One Hot Encoder"
    assert pipeline.random_seed == 0

    parameters = {"One Hot Encoder": {"top_n": 20}, "DBSCAN Clusterer": {"min_samples": 7}}
    pipeline = UnsupervisedPipeline(
        component_graph=["Imputer", "One Hot Encoder", "DBSCAN Clusterer"],
        parameters=parameters,
        custom_name="Custom Pipeline",
        random_seed=42,
    )

    assert pipeline.parameters == {
        "Imputer": {
            "categorical_impute_strategy": "most_frequent",
            "numeric_impute_strategy": "mean",
            "categorical_fill_value": None,
            "numeric_fill_value": None,
        },
        "One Hot Encoder": {
            "top_n": 20,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
        "DBSCAN Clusterer": {"eps": 0.5, "min_samples": 7, "leaf_size": 30, "n_jobs": -1},
    }
    assert pipeline.name == "Custom Pipeline"
    assert pipeline.random_seed == 42


@pytest.mark.parametrize("target_type", ["category", "string", "bool"])
def test_invalid_targets_regression_pipeline(
    breast_cancer_local, wine_local, target_type, dummy_regression_pipeline
):
    X, y = wine_local
    if target_type == "category":
        y = pd.Series(y).astype("category")
    if target_type == "bool":
        X, y = breast_cancer_local
        y = y.map({"malignant": False, "benign": True})
    with pytest.raises(
        ValueError, match="Regression pipeline can only handle numeric target data"
    ):
        dummy_regression_pipeline.fit(X, y)


def test_woodwork_regression_pipeline(diabetes_local, linear_regression_pipeline):
    X, y = diabetes_local
    linear_regression_pipeline.fit(X, y)
    assert not pd.isnull(linear_regression_pipeline.predict(X)).any()


@pytest.mark.parametrize(
    "index",
    [
        list(range(-5, 0)),
        list(range(100, 105)),
        [f"row_{i}" for i in range(5)],
        pd.date_range("2020-09-08", periods=5),
    ],
)
def test_pipeline_transform_and_predict_with_custom_index(
    index,
    linear_regression_pipeline,
):
    X = pd.DataFrame(
        {"categories": [f"cat_{i}" for i in range(5)], "numbers": np.arange(5)},
        index=index,
    )
    X.ww.init(logical_types={"categories": "categorical"})

    y = pd.Series([0, 1.0, 1, 1, 0], index=index)
    linear_regression_pipeline.fit(X, y)
    predictions = linear_regression_pipeline.predict(X)
    assert_index_equal(predictions.index, X.index)
