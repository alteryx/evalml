from itertools import product

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal


@pytest.mark.parametrize("problem_type", ["binary", "multi"])
def test_new_unique_targets_in_score(
    X_y_binary,
    logistic_regression_binary_pipeline,
    X_y_multi,
    logistic_regression_multiclass_pipeline,
    problem_type,
):
    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline
        objective = "Log Loss Binary"
    elif problem_type == "multi":
        X, y = X_y_multi
        pipeline = logistic_regression_multiclass_pipeline
        objective = "Log Loss Multiclass"
    pipeline.fit(X, y)
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        pipeline.score(X, pd.Series([4] * len(y)), [objective])


@pytest.mark.parametrize("num_unique", [1, 2, 3])
@pytest.mark.parametrize("pipeline", ["binary", "multiclass"])
def test_invalid_targets_classification_pipeline(
    num_unique,
    pipeline,
    dummy_binary_pipeline,
    dummy_multiclass_pipeline,
):
    X = pd.DataFrame([i for i in range(30)])

    if num_unique == 1:
        y = pd.Series([1 for i in range(30)])
    elif num_unique == 2:
        y = pd.Series([i % 2 for i in range(30)])
    elif num_unique == 3:
        y = pd.Series([i % 3 for i in range(30)])

    if pipeline == "binary":
        mock_binary_pipeline = dummy_binary_pipeline
        if num_unique in [1, 3]:
            with pytest.raises(
                ValueError,
                match="Binary pipelines require y to have 2 unique classes!",
            ):
                mock_binary_pipeline.fit(X, y)
        else:
            assert mock_binary_pipeline.fit(X, y)
    elif pipeline == "multiclass":
        mock_multi_pipeline = dummy_multiclass_pipeline
        if num_unique in [1, 2]:
            with pytest.raises(
                ValueError,
                match="Multiclass pipelines require y to have 3 or more unique classes!",
            ):
                mock_multi_pipeline.fit(X, y)
        else:
            assert mock_multi_pipeline.fit(X, y)


@pytest.mark.parametrize(
    "problem_type,use_ints",
    product(["binary", "multi"], [True, False]),
)
def test_pipeline_has_classes_property(
    breast_cancer_local,
    wine_local,
    logistic_regression_binary_pipeline,
    logistic_regression_multiclass_pipeline,
    problem_type,
    use_ints,
):
    if problem_type == "binary":
        X, y = breast_cancer_local
        pipeline = logistic_regression_binary_pipeline
        if use_ints:
            y = y.map({"malignant": 0, "benign": 1})
            answer = [0, 1]
        else:
            answer = ["benign", "malignant"]
    elif problem_type == "multi":
        X, y = wine_local
        pipeline = logistic_regression_multiclass_pipeline
        if use_ints:
            y = y.map({"class_0": 0, "class_1": 1, "class_2": 2})
            answer = [0, 1, 2]
        else:
            answer = ["class_0", "class_1", "class_2"]

    # Check that .classes_ is None before fitting
    assert pipeline.classes_ is None

    pipeline.fit(X, y)
    assert pipeline.classes_ == answer


def test_woodwork_classification_pipeline(
    breast_cancer_local,
    logistic_regression_binary_pipeline,
):
    X, y = breast_cancer_local
    mock_pipeline = logistic_regression_binary_pipeline
    mock_pipeline.fit(X, y)
    assert not pd.isnull(mock_pipeline.predict(X)).any()
    assert not pd.isnull(mock_pipeline.predict_proba(X)).any().any()


@pytest.mark.parametrize(
    "index",
    [
        list(range(-5, 0)),
        list(range(100, 105)),
        [f"row_{i}" for i in range(5)],
        pd.date_range("2020-09-08", periods=5),
    ],
)
@pytest.mark.parametrize("problem_type", ["binary", "multi"])
def test_pipeline_transform_and_predict_with_custom_index(
    index,
    problem_type,
    logistic_regression_binary_pipeline,
    logistic_regression_multiclass_pipeline,
):
    X = pd.DataFrame(
        {"categories": [f"cat_{i}" for i in range(5)], "numbers": np.arange(5)},
        index=index,
    )
    X.ww.init(logical_types={"categories": "categorical"})

    if problem_type == "binary":
        y = pd.Series([0, 1, 1, 1, 0], index=index)
        pipeline = logistic_regression_binary_pipeline
    elif problem_type == "multi":
        y = pd.Series([0, 1, 2, 1, 0], index=index)
        pipeline = logistic_regression_multiclass_pipeline
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    predict_proba = pipeline.predict_proba(X)

    assert_index_equal(predictions.index, X.index)
    assert_index_equal(predict_proba.index, X.index)
