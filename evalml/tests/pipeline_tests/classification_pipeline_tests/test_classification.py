from itertools import product

import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_series_equal


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
    breast_cancer_local,
    wine_local,
    logistic_regression_binary_pipeline_class,
    logistic_regression_multiclass_pipeline_class,
    problem_type,
    use_ints,
):
    if problem_type == "binary":
        X, y = breast_cancer_local
        pipeline = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        if use_ints:
            y = y.map({"malignant": 0, "benign": 1})
            answer = pd.Series([0, 1])
        else:
            answer = ww.init_series(pd.Series(["benign", "malignant"]))
    elif problem_type == "multi":
        X, y = wine_local
        pipeline = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        if use_ints:
            y = y.map({"class_0": 0, "class_1": 1, "class_2": 2})
            answer = pd.Series([0, 1, 2])
        else:
            answer = ww.init_series(pd.Series(["class_0", "class_1", "class_2"]))

    # with pytest.raises(
    #     AttributeError, match="Cannot access class names before fitting the pipeline."
    # ):
    #     pipeline.classes_

    pipeline.fit(X, y)
    assert_series_equal(ww.init_series(pd.Series(pipeline.classes_)), pd.Series(answer))


def test_woodwork_classification_pipeline(
    breast_cancer_local, logistic_regression_binary_pipeline_class
):
    X, y = breast_cancer_local
    mock_pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    mock_pipeline.fit(X, y)
    assert not pd.isnull(mock_pipeline.predict(X)).any()
    assert not pd.isnull(mock_pipeline.predict_proba(X)).any().any()
