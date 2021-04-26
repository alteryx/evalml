import pandas as pd
import pytest
from skopt.space import Categorical

from evalml.demos import load_breast_cancer, load_diabetes, load_wine
from evalml.pipelines import RegressionPipeline
from evalml.preprocessing import split_data


def test_regression_init():
    clf = RegressionPipeline(component_graph=["Imputer", "One Hot Encoder", "Random Forest Regressor"])
    assert clf.parameters == {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'mean',
            'categorical_fill_value': None,
            'numeric_fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 10,
            'features_to_encode': None,
            'categories': None,
            'drop': 'if_binary',
            'handle_unknown': 'ignore',
            'handle_missing': 'error'
        },
        'Random Forest Regressor': {
            'n_estimators': 100,
            'max_depth': 6,
            'n_jobs': -1
        }
    }
    assert clf.custom_hyperparameters is None
    assert clf.name == "Random Forest Regressor w/ Imputer + One Hot Encoder"
    assert clf.random_seed == 0
    custom_hyperparameters = {"Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", 'mean'])},
                              "Imputer_1": {"numeric_impute_strategy": Categorical(["median", 'mean'])},
                              "Random Forest Regressor": {"n_estimators": Categorical([50, 100])}}
    parameters = {
        "One Hot Encoder": {
            "top_n": 20
        }
    }
    clf = RegressionPipeline(component_graph=["Imputer", "One Hot Encoder", "Random Forest Regressor"],
                             parameters=parameters,
                             custom_hyperparameters=custom_hyperparameters,
                             custom_name="Custom Pipeline",
                             random_seed=42)

    assert clf.parameters == {
        'Imputer': {
            'categorical_impute_strategy': 'most_frequent',
            'numeric_impute_strategy': 'mean',
            'categorical_fill_value': None,
            'numeric_fill_value': None
        },
        'One Hot Encoder': {
            'top_n': 20,
            'features_to_encode': None,
            'categories': None,
            'drop': 'if_binary',
            'handle_unknown': 'ignore',
            'handle_missing': 'error'
        },
        'Random Forest Regressor': {
            'n_estimators': 100,
            'max_depth': 6,
            'n_jobs': -1
        }
    }
    assert clf.custom_hyperparameters == custom_hyperparameters
    assert clf.name == "Custom Pipeline"
    assert clf.random_seed == 42


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
    X = pd.DataFrame({"a": ["a", "b", "a", "a", "a", "c", "c", "c"], "b": [0, 1, 1, 1, 1, 1, 0, 1]})
    y = pd.Series([0, 0, 0, 1, 0, 1, 0, 0], index=[7, 2, 1, 4, 5, 3, 6, 8])
    x1, x2, y1, y2 = split_data(X, y, problem_type='regression')

    pipeline = RegressionPipeline(component_graph=['Imputer', 'One Hot Encoder', 'Linear Regressor'], parameters={})
    pipeline.fit(x2, y2)
    assert not pd.isnull(pipeline.predict(X).to_series()).any()
