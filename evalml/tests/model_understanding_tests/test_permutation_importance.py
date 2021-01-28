import pytest
from unittest.mock import patch, PropertyMock
import pandas as pd
import numpy as np
from evalml.model_understanding.graphs import calculate_permutation_importance
from evalml.pipelines import BinaryClassificationPipeline, Transformer
from evalml.pipelines.components import TextFeaturizer, OneHotEncoder, DateTimeFeaturizer
from evalml.demos import load_fraud

X, y = load_fraud(100)


class DoubleColumns(Transformer):
    name = "DoubleColumns"

    def __init__(self, random_state=0):
        self._provenance = {}
        super().__init__(parameters={}, component_obj=None, random_state=random_state)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        self._provenance = {col: [f"{col}_doubled"] for col in X.columns}
        new_X = X.assign(**{f"{col}_doubled": 2 * X.loc[:, col] for col in X.columns})
        return new_X.drop(columns=X.columns)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def _get_feature_provenance(self):
        return self._provenance


class LinearPipelineWithDropCols(BinaryClassificationPipeline):
    component_graph = ['Drop Columns Transformer', OneHotEncoder, DateTimeFeaturizer, 'Random Forest Classifier']


class LinearPipelineWithImputer(BinaryClassificationPipeline):
    component_graph = ['Imputer', OneHotEncoder, DateTimeFeaturizer, 'Random Forest Classifier']


class LinearPipelineSameFeatureUsedByTwoComponents(BinaryClassificationPipeline):
    component_graph = ['Imputer', DateTimeFeaturizer, OneHotEncoder, 'Random Forest Classifier']


class LinearPipelineTwoEncoders(BinaryClassificationPipeline):
    component_graph = ['Imputer', DateTimeFeaturizer, OneHotEncoder, OneHotEncoder, "Random Forest Classifier"]


class LinearPipelineWithTextFeatures(BinaryClassificationPipeline):
    component_graph = ['Select Columns Transformer', TextFeaturizer, 'Random Forest Classifier']


class LinearPipelineWithDoubling(BinaryClassificationPipeline):
    component_graph = ['Select Columns Transformer', DoubleColumns, DoubleColumns, DoubleColumns, 'Random Forest Classifier']


class DagTwoEncoders(BinaryClassificationPipeline):
    component_graph = {
            'Imputer': ['Imputer'],
            'SelectNumeric': ["Select Columns Transformer", "Imputer"],
            'SelectCategorical1': ["Select Columns Transformer", "Imputer"],
            'SelectCategorical2': ["Select Columns Transformer", "Imputer"],
            'OHE_1': ['One Hot Encoder', 'SelectCategorical1'],
            'OHE_2': ['One Hot Encoder', 'SelectCategorical2'],
            'DT': ['DateTime Featurization Component', "SelectNumeric"],
            'Estimator': ['Random Forest Classifier', 'DT', 'OHE_1', 'OHE_2'],
        }


class DagReuseFeatures(BinaryClassificationPipeline):
    component_graph = {
            'Imputer': ['Imputer'],
            'SelectNumeric': ["Select Columns Transformer", "Imputer"],
            'SelectCategorical1': ["Select Columns Transformer", "Imputer"],
            'SelectCategorical2': ["Select Columns Transformer", "Imputer"],
            'OHE_1': ['One Hot Encoder', 'SelectCategorical1'],
            'OHE_2': ['One Hot Encoder', 'SelectCategorical2'],
            'DT': ['DateTime Featurization Component', "SelectNumeric"],
            'OHE_3': ['One Hot Encoder', 'DT'],
            'Estimator': ['Random Forest Classifier', 'OHE_3', 'OHE_1', 'OHE_2'],
        }


test_cases = [(LinearPipelineWithDropCols, {"Drop Columns Transformer": {'columns': ['country']}}),
              (LinearPipelineWithImputer, {}),
              (LinearPipelineSameFeatureUsedByTwoComponents, {'DateTime Featurization Component': {'encode_as_categories': True}}),
              (LinearPipelineTwoEncoders, {'One Hot Encoder': {'features_to_encode': ['currency', 'expiration_date', 'provider']},
                                           'One Hot Encoder_2': {'features_to_encode': ['region', 'country']}}),
              (LinearPipelineWithTextFeatures, {'Select Columns Transformer': {'columns': ['amount', 'provider']},
                                                'Text Featurization Component': {'text_columns': ['provider']}}),
              (LinearPipelineWithDoubling, {'Select Columns Transformer': {'columns': ['amount']}}),
              (DagTwoEncoders, {'SelectNumeric': {'columns': ['card_id', 'store_id', 'datetime', 'amount', 'customer_present', 'lat', 'lng']},
                                'SelectCategorical1': {'columns': ['currency', 'expiration_date', 'provider']},
                                'SelectCategorical2': {'columns': ['region', 'country']},
                                'OHE_1': {'features_to_encode': ['currency', 'expiration_date', 'provider']},
                                'OHE_2': {'features_to_encode': ['region', 'country']}}),
              (DagReuseFeatures, {'SelectNumeric': {'columns': ['card_id', 'store_id', 'datetime', 'amount', 'customer_present', 'lat', 'lng']},
                                'SelectCategorical1': {'columns': ['currency', 'expiration_date', 'provider']},
                                'SelectCategorical2': {'columns': ['region', 'country']},
                                'OHE_1': {'features_to_encode': ['currency', 'expiration_date', 'provider']},
                                'OHE_2': {'features_to_encode': ['region', 'country']},
                                'DT': {'encode_as_categories': True}})
              ]


@pytest.mark.parametrize('pipeline_class, parameters', test_cases)
@patch('evalml.pipelines.PipelineBase._supports_fast_permutation_importance', new_callable=PropertyMock)
def test_fast_permutation_importance_matches_sklearn_output(mock_supports_fast_importance, pipeline_class, parameters):

    # Do this to make sure we use the same int as sklearn under the hood
    random_state = np.random.RandomState(0)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    mock_supports_fast_importance.return_value = True
    pipeline = pipeline_class(parameters=parameters)
    pipeline.fit(X, y)
    fast_scores = calculate_permutation_importance(pipeline, X, y, objective='Log Loss Binary',
                                                   random_state=random_seed)

    mock_supports_fast_importance.return_value = False
    slow_scores = calculate_permutation_importance(pipeline, X, y, objective='Log Loss Binary',
                                                   random_state=0)

    pd.testing.assert_frame_equal(fast_scores, slow_scores)


def test_get_permutation_importance_invalid_objective(X_y_regression, linear_regression_pipeline_class):
    X, y = X_y_regression
    pipeline = linear_regression_pipeline_class(parameters={}, random_state=42)
    with pytest.raises(ValueError, match=f"Given objective 'MCC Multiclass' cannot be used with '{pipeline.name}'"):
        calculate_permutation_importance(pipeline, X, y, "mcc multiclass")


@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
def test_get_permutation_importance_binary(X_y_binary, data_type, logistic_regression_binary_pipeline_class,
                                           binary_core_objectives, make_data_type):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
                                                         random_state=42)
    pipeline.fit(X, y)
    for objective in binary_core_objectives:
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_multiclass(X_y_multi, logistic_regression_multiclass_pipeline_class,
                                               multiclass_core_objectives):
    X, y = X_y_multi
    pipeline = logistic_regression_multiclass_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
                                                             random_state=42)
    pipeline.fit(X, y)
    for objective in multiclass_core_objectives:
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_regression(linear_regression_pipeline_class, regression_core_objectives):
    X = pd.DataFrame([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    y = pd.Series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    pipeline = linear_regression_pipeline_class(parameters={"Linear Regressor": {"n_jobs": 1}},
                                                random_state=42)
    pipeline.fit(X, y)

    for objective in regression_core_objectives:
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_correlated_features(logistic_regression_binary_pipeline_class):
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["correlated"] = y * 2
    X["not correlated"] = [-1, -1, -1, 0]
    y = y.astype(bool)
    pipeline = logistic_regression_binary_pipeline_class(parameters={}, random_state=42)
    pipeline.fit(X, y)
    importance = calculate_permutation_importance(pipeline, X, y, objective="Log Loss Binary", random_state=0)
    assert list(importance.columns) == ["feature", "importance"]
    assert not importance.isnull().all().all()
    correlated_importance_val = importance["importance"][importance.index[importance["feature"] == "correlated"][0]]
    not_correlated_importance_val = importance["importance"][importance.index[importance["feature"] == "not correlated"][0]]
    assert correlated_importance_val > not_correlated_importance_val