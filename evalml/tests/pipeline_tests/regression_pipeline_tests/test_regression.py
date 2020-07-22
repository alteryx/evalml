import pandas as pd
import pytest

from evalml.demos import load_breast_cancer, load_wine


@pytest.mark.parametrize("target_type", ["categorical", "string", "bool"])
def test_invalid_targets_regression_pipeline(target_type, dummy_regression_pipeline_class):
    X, y = load_wine()
    if target_type == "categorical":
        y = pd.Categorical(y)
    if target_type == "bool":
        y = y.map({"malignant": False, "benign": True})
        X, y = load_breast_cancer()
    mock_regression_pipeline = dummy_regression_pipeline_class(parameters={})
    with pytest.raises(ValueError, match="Regression pipeline cannot handle targets with dtype"):
        mock_regression_pipeline.fit(X, y)
