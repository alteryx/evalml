import pandas as pd
import pytest
import woodwork as ww

from evalml.demos import load_breast_cancer, load_diabetes, load_wine


@pytest.mark.parametrize("target_type", ["category", "string", "bool"])
def test_invalid_targets_regression_pipeline(target_type, dummy_regression_pipeline_class):
    X, y = load_wine()
    if target_type == "category":
        y = pd.Series(y).astype("category")
    if target_type == "bool":
        X, y = load_breast_cancer()
        y = y.map({"malignant": False, "benign": True})
    mock_regression_pipeline = dummy_regression_pipeline_class(parameters={})
    with pytest.raises(ValueError, match="Regression pipeline cannot handle targets with dtype"):
        mock_regression_pipeline.fit(X, y)


def test_woodwork_regression_pipeline(linear_regression_pipeline_class):
    X, y = load_diabetes()
    X = ww.DataTable(X)
    y = ww.DataColumn(y)
    mock_regression_pipeline = linear_regression_pipeline_class(parameters={})
    mock_regression_pipeline.fit(X, y)
    assert not pd.isnull(mock_regression_pipeline.predict(X)).any()
