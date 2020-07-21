import pandas as pd
import pytest

from evalml.demos import load_breast_cancer, load_wine


@pytest.mark.parametrize("target_type", ["categorical", "string", "bool"])
def test_invalid_targets_regression_pipeline(target_type, dummy_regression_pipeline_class):
    X, y = load_wine()
    if target_type == "categorical":
        y = pd.Categorical(y)
    elif target_type == "int":
        unique_vals = y.unique()
        y = y.map({unique_vals[i]: int(i) for i in range(len(unique_vals))})
    elif target_type == "float":
        unique_vals = y.unique()
        y = y.map({unique_vals[i]: float(i) for i in range(len(unique_vals))})
    if target_type == "bool":
        y = y.map({"malignant": False, "benign": True})
        X, y = load_breast_cancer()
    mock_regression_pipeline = dummy_regression_pipeline_class(parameters={})
    with pytest.raises(ValueError, match="Regression pipeline cannot handle targets with dtype"):
        mock_regression_pipeline.fit(X, y)
