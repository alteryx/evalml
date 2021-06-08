from itertools import product
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from evalml.demos import load_breast_cancer, load_wine
from evalml.objectives.utils import get_core_objectives


@pytest.mark.parametrize("problem_type", ["binary", "multi"])
def test_new_unique_targets_in_score(
    X_y_binary,
    logistic_regression_binary_pipeline_class,
    X_y_multi,
    logistic_regression_multiclass_pipeline_class,
    problem_type,
):
    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        objective = "Log Loss Binary"
    elif problem_type == "multi":
        X, y = X_y_multi
        pipeline = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        objective = "Log Loss Multiclass"
    pipeline.fit(X, y)
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        pipeline.score(X, pd.Series([4] * len(y)), [objective])


@pytest.mark.parametrize(
    "problem_type,use_ints", product(["binary", "multi"], [True, False])
)
def test_pipeline_has_classes_property(
    logistic_regression_binary_pipeline_class,
    logistic_regression_multiclass_pipeline_class,
    problem_type,
    use_ints,
):
    if problem_type == "binary":
        X, y = load_breast_cancer()
        pipeline = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        if use_ints:
            y = y.map({"malignant": 0, "benign": 1})
            answer = [0, 1]
        else:
            answer = ["benign", "malignant"]
    elif problem_type == "multi":
        X, y = load_wine()
        pipeline = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        if use_ints:
            y = y.map({"class_0": 0, "class_1": 1, "class_2": 2})
            answer = [0, 1, 2]
        else:
            answer = ["class_0", "class_1", "class_2"]

    with pytest.raises(
        AttributeError, match="Cannot access class names before fitting the pipeline."
    ):
        pipeline.classes_

    pipeline.fit(X, y)
    assert_series_equal(pd.Series(pipeline.classes_), pd.Series(answer))


def test_woodwork_classification_pipeline(logistic_regression_binary_pipeline_class):
    X, y = load_breast_cancer()
    mock_pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    mock_pipeline.fit(X, y)
    assert not pd.isnull(mock_pipeline.predict(X)).any()
    assert not pd.isnull(mock_pipeline.predict_proba(X)).any().any()


@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.predict")
def test_classification_pipeline_scoring_with_nan_in_target(
    mock_predict, mock_predict_proba, logistic_regression_binary_pipeline_class
):
    X, y = load_breast_cancer()
    mock_pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )

    mock_predict.return_value = pd.Series(np.resize([0, 1, np.nan], len(y)))
    mock_predict_proba_df = pd.DataFrame({"benign": pd.Series(np.resize([0.4, 0.6], len(y))),
    "malignant": pd.Series(np.resize([0.6, 0.4], len(y)))
    })
    mock_predict_proba_df.loc[0, "benign"] = np.nan
    mock_predict_proba_df.loc[2, "malignant"] = np.nan
    mock_rows_with_nan = list(mock_predict_proba_df[mock_predict_proba_df.isnull().any(axis=1)].index)
    assert mock_rows_with_nan == [0, 2]
    mock_predict_proba.return_value = mock_predict_proba_df

    mock_pipeline.fit(X, y)
    assert pd.isnull(mock_pipeline.predict(X)).any()
    assert pd.isnull(mock_pipeline.predict_proba(X)).any().any()
    scores = mock_pipeline.score(X, y, get_core_objectives("binary"))
    for score in scores.values():
        assert not np.isnan(score)

    mock_predict_proba.assert_called()
    mock_predict.assert_called()