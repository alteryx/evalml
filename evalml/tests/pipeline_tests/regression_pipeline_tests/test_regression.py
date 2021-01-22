import pandas as pd
import pytest

from evalml.demos import load_breast_cancer, load_diabetes, load_wine
from evalml.pipelines import RegressionPipeline
from evalml.preprocessing import split_data


@pytest.mark.parametrize("target_type", ["category", "string", "bool"])
def test_invalid_targets_regression_pipeline(target_type, dummy_regression_pipeline_class):
    X, y = load_wine(return_pandas=True)
    if target_type == "category":
        y = pd.Series(y).astype("category")
    if target_type == "bool":
        X, y = load_breast_cancer(return_pandas=True)
        y = y.map({"malignant": False, "benign": True})
    mock_regression_pipeline = dummy_regression_pipeline_class(parameters={})
    with pytest.raises(ValueError, match="Regression pipeline can only handle numeric target data"):
        mock_regression_pipeline.fit(X, y)


def test_woodwork_regression_pipeline(linear_regression_pipeline_class):
    X, y = load_diabetes()
    regression_pipeline = linear_regression_pipeline_class(parameters={'Linear Regressor': {'n_jobs': 1}})
    regression_pipeline.fit(X, y)
    assert not pd.isnull(regression_pipeline.predict(X).to_series()).any()


def test_custom_indices():
    class MyPipeline(RegressionPipeline):
        component_graph = ['Imputer', 'One Hot Encoder', 'Linear Regressor']
        custom_name = "My Pipeline"

    X = pd.DataFrame({"a": ["a", "b", "a", "a", "a", "c", "c", "c"], "b": [0, 1, 1, 1, 1, 1, 0, 1]})
    y = pd.Series([0, 0, 0, 1, 0, 1, 0, 0], index=[7, 2, 1, 4, 5, 3, 6, 8])

    x1, x2, y1, y2 = split_data(X, y, problem_type='regression')
    pipeline = MyPipeline({})
    pipeline.fit(x2, y2)
    assert not pd.isnull(pipeline.predict(X).to_series()).any()
