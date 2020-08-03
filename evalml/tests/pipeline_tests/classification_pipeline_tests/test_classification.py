import pandas as pd
import pytest

from evalml.demos import load_breast_cancer, load_wine


@pytest.mark.parametrize("problem_type", ["binary", "multi"])
def test_new_unique_targets_in_score(X_y_binary, logistic_regression_binary_pipeline_class,
                                     X_y_multi, logistic_regression_multiclass_pipeline_class, problem_type):
    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline_class(parameters={})
        objective = 'log_loss_binary'
    elif problem_type == "multi":
        X, y = X_y_multi
        pipeline = logistic_regression_multiclass_pipeline_class(parameters={})
        objective = 'log_loss_multi'
    pipeline.fit(X, y)
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        pipeline.score(X, pd.Series([4] * len(y)), [objective])


@pytest.mark.parametrize("problem_type", ["binary", "multi"])
def test_pipeline_has_classes_property(logistic_regression_binary_pipeline_class,
                                       logistic_regression_multiclass_pipeline_class, problem_type):
    if problem_type == "binary":
        X, y = load_breast_cancer()
        pipeline = logistic_regression_binary_pipeline_class(parameters={})
        answer = ["benign", "malignant"]
    elif problem_type == "multi":
        X, y = load_wine()
        pipeline = logistic_regression_multiclass_pipeline_class(parameters={})
        answer = ["class_0", "class_1", "class_2"]

    pipeline.fit(X, y)
    pd.testing.assert_series_equal(pd.Series(pipeline._classes), pd.Series(answer))
