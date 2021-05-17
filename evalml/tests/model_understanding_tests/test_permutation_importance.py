from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.demos import load_fraud
from evalml.model_understanding.graphs import calculate_permutation_importance
from evalml.pipelines import BinaryClassificationPipeline, Transformer
from evalml.pipelines.components import (
    PCA,
    DateTimeFeaturizer,
    DFSTransformer,
    OneHotEncoder,
    TextFeaturizer
)
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types


class DoubleColumns(Transformer):
    """Custom transformer for testing permutation importance implementation.

    We don't have any transformers that create features that you can repeatedly "stack" on the previous output.
    That being said, I want to test that our implementation can handle that case in the event we add a transformer like
    that in the future.
    """
    name = "DoubleColumns"
    hyperparameter_ranges = {}

    def __init__(self, drop_old_columns=True, random_seed=0):
        self._provenance = {}
        self.drop_old_columns = drop_old_columns
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        self._provenance = {col: [f"{col}_doubled"] for col in X.columns}
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        new_X = X.assign(**{f"{col}_doubled": 2 * X.loc[:, col] for col in X.columns})
        if self.drop_old_columns:
            new_X = new_X.drop(columns=X.columns)
        return infer_feature_types(new_X)

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
    component_graph = ['Imputer', 'Drop Columns Transformer', TextFeaturizer, OneHotEncoder, 'Random Forest Classifier']


class LinearPipelineWithTextFeaturizerNoTextFeatures(LinearPipelineWithTextFeatures):
    """Testing a pipeline with TextFeaturizer but no text features."""


class LinearPipelineWithDoubling(BinaryClassificationPipeline):
    component_graph = ['Select Columns Transformer', DoubleColumns, DoubleColumns, DoubleColumns, 'Random Forest Classifier']


class LinearPipelineWithTargetEncoderAndOHE(BinaryClassificationPipeline):
    component_graph = ['Imputer', DateTimeFeaturizer, OneHotEncoder, 'Target Encoder', "Random Forest Classifier"]


class LinearPipelineCreateFeatureThenDropIt(BinaryClassificationPipeline):
    component_graph = ['Select Columns Transformer', DoubleColumns, 'Drop Columns Transformer', 'Random Forest Classifier']


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
              (LinearPipelineWithTextFeatures, {'Drop Columns Transformer': {'columns': ['datetime']}}),
              (LinearPipelineWithTextFeaturizerNoTextFeatures, {'Drop Columns Transformer': {'columns': ['datetime']}}),
              (LinearPipelineWithDoubling, {'Select Columns Transformer': {'columns': ['amount']}}),
              (LinearPipelineWithDoubling, {'Select Columns Transformer': {'columns': ['amount']},
                                            'DoubleColumns': {'drop_old_columns': False}}),
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
                                  'DT': {'encode_as_categories': True}}),
              (LinearPipelineWithTargetEncoderAndOHE, {'One Hot Encoder': {'features_to_encode': ['currency', 'expiration_date', 'provider']},
                                                       'Target Encoder': {'cols': ['region', 'country']}}),
              (LinearPipelineCreateFeatureThenDropIt, {'Select Columns Transformer': {'columns': ['amount']},
                                                       'DoubleColumns': {'drop_old_columns': False},
                                                       'Drop Columns Transformer': {'columns': ['amount_doubled']}})
              ]


@pytest.mark.parametrize('pipeline_class, parameters', test_cases)
@patch('evalml.pipelines.PipelineBase._supports_fast_permutation_importance', new_callable=PropertyMock)
def test_fast_permutation_importance_matches_sklearn_output(mock_supports_fast_importance, pipeline_class, parameters,
                                                            has_minimal_dependencies):
    if has_minimal_dependencies and pipeline_class == LinearPipelineWithTargetEncoderAndOHE:
        pytest.skip("Skipping test_fast_permutation_importance_matches_sklearn_output for target encoder cause "
                    "dependency not installed.")
    X, y = load_fraud(100)

    if pipeline_class == LinearPipelineWithTextFeatures:
        X = X.set_types(logical_types={'provider': 'NaturalLanguage'})

    # Do this to make sure we use the same int as sklearn under the hood
    random_state = np.random.RandomState(0)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    mock_supports_fast_importance.return_value = True
    parameters['Random Forest Classifier'] = {'n_jobs': 1}
    pipeline = pipeline_class(pipeline_class.component_graph, parameters=parameters)
    pipeline.fit(X, y)
    fast_scores = calculate_permutation_importance(pipeline, X, y, objective='Log Loss Binary',
                                                   random_seed=random_seed)
    mock_supports_fast_importance.return_value = False
    slow_scores = calculate_permutation_importance(pipeline, X, y, objective='Log Loss Binary',
                                                   random_seed=0)
    pd.testing.assert_frame_equal(fast_scores, slow_scores)


class PipelineWithDimReduction(BinaryClassificationPipeline):
    component_graph = [PCA, 'Logistic Regression Classifier']

    def __init__(self, parameters, random_seed=0):
        super().__init__(self.component_graph, parameters=parameters, custom_hyperparameters=None, random_seed=random_seed)


class EnsembleDag(BinaryClassificationPipeline):
    component_graph = {
        'Imputer_1': ['Imputer'],
        'Imputer_2': ['Imputer'],
        'OHE_1': ['One Hot Encoder', 'Imputer_1'],
        'OHE_2': ['One Hot Encoder', 'Imputer_2'],
        'DT_1': ['DateTime Featurization Component', 'OHE_1'],
        'DT_2': ['DateTime Featurization Component', 'OHE_2'],
        'Estimator_1': ['Random Forest Classifier', 'DT_1'],
        'Estimator_2': ['Extra Trees Classifier', 'DT_2'],
        'Ensembler': ['Logistic Regression Classifier', 'Estimator_1', 'Estimator_2']
    }

    def __init__(self, parameters, random_seed=0):
        super().__init__(self.component_graph, parameters=parameters, custom_hyperparameters=None, random_seed=random_seed)


class PipelineWithDFS(BinaryClassificationPipeline):
    component_graph = [DFSTransformer, 'Logistic Regression Classifier']

    def __init__(self, parameters, random_seed=0):
        super().__init__(self.component_graph, parameters=parameters, custom_hyperparameters=None, random_seed=random_seed)


class PipelineWithCustomComponent(BinaryClassificationPipeline):
    component_graph = [DoubleColumns, 'Logistic Regression Classifier']

    def __init__(self, parameters, random_seed=0):
        super().__init__(self.component_graph, parameters=parameters, custom_hyperparameters=None, random_seed=random_seed)


class StackedEnsemblePipeline(BinaryClassificationPipeline):
    component_graph = ['Stacked Ensemble Classifier']

    def __init__(self, parameters, random_seed=0):
        super().__init__(self.component_graph, parameters=parameters, custom_hyperparameters=None, random_seed=random_seed)


pipelines_that_do_not_support_fast_permutation_importance = [PipelineWithDimReduction,
                                                             PipelineWithDFS,
                                                             PipelineWithCustomComponent,
                                                             EnsembleDag, StackedEnsemblePipeline]


@pytest.mark.parametrize('pipeline_class', pipelines_that_do_not_support_fast_permutation_importance)
def test_supports_fast_permutation_importance(pipeline_class):
    params = {'Stacked Ensemble Classifier': {'input_pipelines': [PipelineWithDFS({})]}}
    assert not pipeline_class(params)._supports_fast_permutation_importance


def test_get_permutation_importance_invalid_objective(X_y_regression, linear_regression_pipeline_class):
    X, y = X_y_regression
    pipeline = linear_regression_pipeline_class(parameters={}, random_seed=42)
    with pytest.raises(ValueError, match=f"Given objective 'MCC Multiclass' cannot be used with '{pipeline.name}'"):
        calculate_permutation_importance(pipeline, X, y, "mcc multiclass")


@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
def test_get_permutation_importance_binary(X_y_binary, data_type, logistic_regression_binary_pipeline_class,
                                           binary_core_objectives, make_data_type):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
                                                         random_seed=42)
    pipeline.fit(X, y)
    for objective in binary_core_objectives:
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_multiclass(X_y_multi, logistic_regression_multiclass_pipeline_class,
                                               multiclass_core_objectives):
    X, y = X_y_multi
    pipeline = logistic_regression_multiclass_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
                                                             random_seed=42)
    pipeline.fit(X, y)
    for objective in multiclass_core_objectives:
        permutation_importance = calculate_permutation_importance(pipeline, X, y, objective)
        assert list(permutation_importance.columns) == ["feature", "importance"]
        assert not permutation_importance.isnull().all().all()


def test_get_permutation_importance_regression(linear_regression_pipeline_class, regression_core_objectives):
    X = pd.DataFrame([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    y = pd.Series([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    pipeline = linear_regression_pipeline_class(parameters={"Linear Regressor": {"n_jobs": 1}},
                                                random_seed=42)
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
    pipeline = logistic_regression_binary_pipeline_class(parameters={}, random_seed=42)
    pipeline.fit(X, y)
    importance = calculate_permutation_importance(pipeline, X, y, objective="Log Loss Binary", random_seed=0)
    assert list(importance.columns) == ["feature", "importance"]
    assert not importance.isnull().all().all()
    correlated_importance_val = importance["importance"][importance.index[importance["feature"] == "correlated"][0]]
    not_correlated_importance_val = importance["importance"][importance.index[importance["feature"] == "not correlated"][0]]
    assert correlated_importance_val > not_correlated_importance_val


def test_undersampler(X_y_binary):
    """Smoke test to enable hotfix for 0.24.0.  Prior to the 0.24.0 hotfix, this test will
    generate a ValueError within calculate_permutation_importance.

    TODO: Remove with github issue #2273
    """
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series(y)
    pipeline = BinaryClassificationPipeline(component_graph=["Undersampler", "Elastic Net Classifier"])
    pipeline.fit(X=X, y=y)
    pipeline.predict(X)
    test = calculate_permutation_importance(pipeline, X, y, objective="Log Loss Binary")
    assert test is not None
