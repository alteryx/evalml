import pandas as pd
import pytest


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
