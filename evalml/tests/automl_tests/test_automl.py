import os
import warnings
from collections import OrderedDict
from itertools import product
from unittest.mock import MagicMock, PropertyMock, patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn.model_selection import KFold, StratifiedKFold
from skopt.space import Categorical, Integer, Real

from evalml import AutoMLSearch
from evalml.automl.callbacks import (
    log_error_callback,
    raise_error_callback,
    silent_error_callback
)
from evalml.automl.utils import (
    _LARGE_DATA_PERCENT_VALIDATION,
    _LARGE_DATA_ROW_THRESHOLD,
    get_default_primary_search_objective
)
from evalml.demos import load_breast_cancer, load_wine
from evalml.exceptions import (
    AutoMLSearchException,
    PipelineNotFoundError,
    PipelineNotYetFittedError,
    PipelineScoreError
)
from evalml.model_family import ModelFamily
from evalml.objectives import (
    F1,
    BinaryClassificationObjective,
    CostBenefitMatrix,
    FraudCost,
    RegressionObjective
)
from evalml.objectives.utils import (
    get_all_objective_names,
    get_core_objectives,
    get_non_core_objectives,
    get_objective
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    Estimator,
    MulticlassClassificationPipeline,
    PipelineBase,
    RegressionPipeline,
    StackedEnsembleClassifier
)
from evalml.pipelines.components.utils import (
    allowed_model_families,
    get_estimators
)
from evalml.pipelines.utils import make_pipeline
from evalml.preprocessing import TrainingValidationSplit, split_data
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_classification,
    is_time_series
)
from evalml.tuners import NoParamsException, RandomSearchTuner


@pytest.mark.parametrize("automl_type,objective",
                         zip([ProblemTypes.REGRESSION, ProblemTypes.MULTICLASS, ProblemTypes.BINARY, ProblemTypes.BINARY],
                             ['R2', 'log loss multiclass', 'log loss binary', 'F1']))
def test_search_results(X_y_regression, X_y_binary, X_y_multi, automl_type, objective):
    expected_cv_data_keys = {'all_objective_scores', "mean_cv_score", 'binary_classification_threshold'}
    if automl_type == ProblemTypes.REGRESSION:
        expected_pipeline_class = RegressionPipeline
        X, y = X_y_regression
    elif automl_type == ProblemTypes.BINARY:
        expected_pipeline_class = BinaryClassificationPipeline
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        expected_pipeline_class = MulticlassClassificationPipeline
        X, y = X_y_multi

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, optimize_thresholds=False, objective=objective, max_iterations=2, n_jobs=1)
    automl.search()
    assert automl.results.keys() == {'pipeline_results', 'search_order'}
    assert automl.results['search_order'] == [0, 1]
    assert len(automl.results['pipeline_results']) == 2
    for pipeline_id, results in automl.results['pipeline_results'].items():
        assert results.keys() == {'id', 'pipeline_name', 'pipeline_class', 'pipeline_summary', 'parameters', "mean_cv_score",
                                  "standard_deviation_cv_score", 'high_variance_cv', 'training_time',
                                  'cv_data', 'percent_better_than_baseline_all_objectives',
                                  'percent_better_than_baseline', 'validation_score'}
        assert results['id'] == pipeline_id
        assert isinstance(results['pipeline_name'], str)
        assert issubclass(results['pipeline_class'], expected_pipeline_class)
        assert isinstance(results['pipeline_summary'], str)
        assert isinstance(results['parameters'], dict)
        assert isinstance(results["mean_cv_score"], float)
        assert isinstance(results['high_variance_cv'], bool)
        assert isinstance(results['cv_data'], list)
        for cv_result in results['cv_data']:
            assert cv_result.keys() == expected_cv_data_keys
            if objective == 'F1':
                assert cv_result['binary_classification_threshold'] == 0.5
            else:
                assert cv_result['binary_classification_threshold'] is None
            all_objective_scores = cv_result["all_objective_scores"]
            for score in all_objective_scores.values():
                assert score is not None
        assert automl.get_pipeline(pipeline_id).parameters == results['parameters']
        assert results['validation_score'] == pd.Series([fold["mean_cv_score"] for fold in results['cv_data']])[0]
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)
    assert np.all(automl.rankings.dtypes == pd.Series(
        [np.dtype('int64'), np.dtype('O'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('O')],
        index=['id', 'pipeline_name', "mean_cv_score", "standard_deviation_cv_score", "validation_score", 'percent_better_than_baseline', 'high_variance_cv', 'parameters']))
    assert np.all(automl.full_rankings.dtypes == pd.Series(
        [np.dtype('int64'), np.dtype('O'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('O')],
        index=['id', 'pipeline_name', "mean_cv_score", "standard_deviation_cv_score", "validation_score", 'percent_better_than_baseline', 'high_variance_cv', 'parameters']))


@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_pipeline_limits(mock_fit_binary, mock_score_binary,
                         mock_fit_multi, mock_score_multi,
                         mock_fit_regression, mock_score_regression,
                         automl_type, caplog,
                         X_y_binary, X_y_multi, X_y_regression):
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression

    mock_score_binary.return_value = {'Log Loss Binary': 1.0}
    mock_score_multi.return_value = {'Log Loss Multiclass': 1.0}
    mock_score_regression.return_value = {'R2': 1.0}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, max_iterations=1)
    automl.search()
    out = caplog.text
    assert "Searching up to 1 pipelines. " in out
    assert len(automl.results['pipeline_results']) == 1

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, max_time=1)
    automl.search()
    out = caplog.text
    assert "Will stop searching for new pipelines after 1 seconds" in out
    assert len(automl.results['pipeline_results']) >= 1

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, max_time=1, max_iterations=5)
    automl.search()
    out = caplog.text
    assert "Searching up to 5 pipelines. " in out
    assert "Will stop searching for new pipelines after 1 seconds" in out
    assert len(automl.results['pipeline_results']) <= 5

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type)
    automl.search()
    out = caplog.text
    assert "Using default limit of max_batches=1." in out
    assert "Searching up to 1 batches for a total of" in out
    assert len(automl.results['pipeline_results']) > 5

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, max_time=1e-16)
    automl.search()
    out = caplog.text
    assert "Will stop searching for new pipelines after 0 seconds" in out
    # search will always run at least one pipeline
    assert len(automl.results['pipeline_results']) >= 1


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_pipeline_fit_raises(mock_fit, X_y_binary, caplog):
    msg = 'all your model are belong to us'
    mock_fit.side_effect = Exception(msg)
    X, y = X_y_binary
    # Don't train the best pipeline, since this test mocks the pipeline.fit() method and causes it to raise an exception,
    # which we don't want to raise while fitting the best pipeline.
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1, train_best_pipeline=False)
    automl.search()
    out = caplog.text
    assert 'Exception during automl search' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1

    cv_scores_all = pipeline_results[0].get('cv_data', {})
    for cv_scores in cv_scores_all:
        for name, score in cv_scores['all_objective_scores'].items():
            if name in ['# Training', '# Validation']:
                assert score > 0
            else:
                assert np.isnan(score)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
def test_pipeline_score_raises(mock_score, X_y_binary, caplog):
    msg = 'all your model are belong to us'
    mock_score.side_effect = Exception(msg)
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1, n_jobs=1)
    automl.search()
    out = caplog.text
    assert 'Exception during automl search' in out
    assert 'All scores will be replaced with nan.' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0]["cv_data"][0]["all_objective_scores"]
    objective_scores = {o.name: cv_scores_all[o.name] for o in [automl.objective] + automl.additional_objectives}

    assert np.isnan(list(objective_scores.values())).all()


@patch('evalml.objectives.AUC.score')
def test_objective_score_raises(mock_score, X_y_binary, caplog):
    msg = 'all your model are belong to us'
    mock_score.side_effect = Exception(msg)
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1, n_jobs=1)
    automl.search()
    out = caplog.text

    assert msg in out
    pipeline_results = automl.results.get('pipeline_results')
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0].get('cv_data')
    scores = cv_scores_all[0]['all_objective_scores']
    auc_score = scores.pop('AUC')
    assert np.isnan(auc_score)
    assert not np.isnan(list(scores.values())).any()


def test_rankings(X_y_binary, X_y_regression):
    X, y = X_y_binary
    model_families = ['random_forest']
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', allowed_model_families=model_families,
                          max_iterations=3, n_jobs=1)
    automl.search()
    assert len(automl.full_rankings) == 3
    assert len(automl.rankings) == 2

    X, y = X_y_regression
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_model_families=model_families, max_iterations=3,
                          n_jobs=1)
    automl.search()
    assert len(automl.full_rankings) == 3
    assert len(automl.rankings) == 2


@patch('evalml.objectives.BinaryClassificationObjective.optimize_threshold')
@patch('evalml.pipelines.BinaryClassificationPipeline._encode_targets', side_effect=lambda y: y)
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_str_search(mock_fit, mock_score, mock_predict_proba, mock_encode_targets, mock_optimize_threshold, X_y_binary):
    def _dummy_callback(param1, param2, param3):
        return None

    X, y = X_y_binary
    search_params = {
        'problem_type': 'binary',
        'objective': 'F1',
        'max_time': 100,
        'max_iterations': 5,
        'patience': 2,
        'tolerance': 0.5,
        'allowed_model_families': ['random_forest', 'linear_model'],
        'data_splitter': StratifiedKFold(n_splits=5),
        'tuner_class': RandomSearchTuner,
        'start_iteration_callback': _dummy_callback,
        'add_result_callback': None,
        'additional_objectives': ['Precision', 'AUC'],
        'n_jobs': 2,
        'optimize_thresholds': True
    }

    param_str_reps = {
        'Objective': search_params['objective'],
        'Max Time': search_params['max_time'],
        'Max Iterations': search_params['max_iterations'],
        'Allowed Pipelines': [],
        'Patience': search_params['patience'],
        'Tolerance': search_params['tolerance'],
        'Data Splitting': 'StratifiedKFold(n_splits=5, random_state=None, shuffle=False)',
        'Tuner': 'RandomSearchTuner',
        'Start Iteration Callback': '_dummy_callback',
        'Add Result Callback': None,
        'Additional Objectives': search_params['additional_objectives'],
        'Random Seed': 0,
        'n_jobs': search_params['n_jobs'],
        'Optimize Thresholds': search_params['optimize_thresholds']
    }

    automl = AutoMLSearch(X_train=X, y_train=y, **search_params)
    mock_score.return_value = {automl.objective.name: 1.0}
    mock_optimize_threshold.return_value = 0.62
    str_rep = str(automl)
    for param, value in param_str_reps.items():
        if isinstance(value, (tuple, list)):
            assert f"{param}" in str_rep
            for item in value:
                s = f"\t{str(item)}" if isinstance(value, list) else f"{item}"
                assert s in str_rep
        else:
            assert f"{param}: {str(value)}" in str_rep
    assert "Search Results" not in str_rep

    mock_score.return_value = {automl.objective.name: 1.0}
    mock_predict_proba.return_value = ww.DataTable(pd.DataFrame([[1.0, 0.0], [0.0, 1.0]]))
    automl.search()
    mock_fit.assert_called()
    mock_score.assert_called()
    mock_predict_proba.assert_called()
    mock_optimize_threshold.assert_called()

    str_rep = str(automl)
    assert "Search Results:" in str_rep
    assert automl.rankings.drop(['parameters'], axis='columns').to_string() in str_rep


def test_automl_str_no_param_search(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary')

    param_str_reps = {
        'Objective': 'Log Loss Binary',
        'Max Time': 'None',
        'Max Iterations': 'None',
        'Allowed Pipelines': [],
        'Patience': 'None',
        'Tolerance': '0.0',
        'Data Splitting': 'StratifiedKFold(n_splits=5, random_state=None, shuffle=False)',
        'Tuner': 'SKOptTuner',
        'Additional Objectives': [
            'AUC',
            'Accuracy Binary',
            'Balanced Accuracy Binary',
            'F1',
            'MCC Binary',
            'Precision'],
        'Start Iteration Callback': 'None',
        'Add Result Callback': 'None',
        'Random Seed': 0,
        'n_jobs': '-1',
        'Optimize Thresholds': 'False'
    }

    str_rep = str(automl)
    for param, value in param_str_reps.items():
        assert f"{param}" in str_rep
        if isinstance(value, list):
            value = "\n".join(["\t{}".format(item) for item in value])
            assert value in str_rep
    assert "Search Results" not in str_rep


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_feature_selection(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    class MockFeatureSelectionPipeline(BinaryClassificationPipeline):
        component_graph = ['RF Classifier Select From Model', 'Logistic Regression Classifier']

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

        def fit(self, X, y):
            """Mock fit, noop"""

    allowed_pipelines = [MockFeatureSelectionPipeline({})]
    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=2,
                          start_iteration_callback=start_iteration_callback, allowed_pipelines=allowed_pipelines)
    automl.search()

    assert start_iteration_callback.call_count == 2
    proposed_parameters = start_iteration_callback.call_args_list[1][0][1]
    assert proposed_parameters.keys() == {'RF Classifier Select From Model', 'Logistic Regression Classifier'}
    assert proposed_parameters['RF Classifier Select From Model']['number_features'] == X.shape[1]


@patch('evalml.tuners.random_search_tuner.RandomSearchTuner.is_search_space_exhausted')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_tuner_exception(mock_fit, mock_score, mock_is_search_space_exhausted, X_y_binary):
    mock_score.return_value = {'Log Loss Binary': 1.0}
    X, y = X_y_binary
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    mock_is_search_space_exhausted.side_effect = NoParamsException(error_text)
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', objective="R2", tuner_class=RandomSearchTuner, max_iterations=10)
    with pytest.raises(NoParamsException, match=error_text):
        automl.search()


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_algorithm(mock_fit, mock_score, mock_algo_next_batch, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}
    mock_algo_next_batch.side_effect = StopIteration("that's all, folks")
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=5)
    automl.search()
    mock_fit.assert_called()
    mock_score.assert_called()
    assert mock_algo_next_batch.call_count == 1
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    assert pipeline_results[0].get("mean_cv_score") == 1.0


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.__init__')
def test_automl_allowed_pipelines_algorithm(mock_algo_init, dummy_binary_pipeline_class, X_y_binary):
    mock_algo_init.side_effect = Exception('mock algo init')
    X, y = X_y_binary

    allowed_pipelines = [dummy_binary_pipeline_class({})]
    with pytest.raises(Exception, match='mock algo init'):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', allowed_pipelines=allowed_pipelines, max_iterations=10)
    assert mock_algo_init.call_count == 1
    _, kwargs = mock_algo_init.call_args
    assert kwargs['max_iterations'] == 10
    assert kwargs['allowed_pipelines'] == allowed_pipelines

    allowed_model_families = [ModelFamily.RANDOM_FOREST]
    with pytest.raises(Exception, match='mock algo init'):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', allowed_model_families=allowed_model_families, max_iterations=1)
    assert mock_algo_init.call_count == 2
    _, kwargs = mock_algo_init.call_args
    assert kwargs['max_iterations'] == 1
    for actual, expected in zip(kwargs['allowed_pipelines'], [make_pipeline(X, y, estimator, ProblemTypes.BINARY) for estimator in get_estimators(ProblemTypes.BINARY, model_families=allowed_model_families)]):
        assert actual.parameters == expected.parameters


def test_automl_serialization(X_y_binary, tmpdir):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), 'automl.pkl')
    num_max_iterations = 5
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=num_max_iterations, n_jobs=1)
    automl.search()
    automl.save(path)
    loaded_automl = automl.load(path)

    for i in range(num_max_iterations):
        assert automl.get_pipeline(i).__class__ == loaded_automl.get_pipeline(i).__class__
        assert automl.get_pipeline(i).parameters == loaded_automl.get_pipeline(i).parameters

        for id_, pipeline_results in automl.results['pipeline_results'].items():
            loaded_ = loaded_automl.results['pipeline_results'][id_]
            for name in pipeline_results:
                # Use np to check percent_better_than_baseline because of (possible) nans
                if name == 'percent_better_than_baseline_all_objectives':
                    for objective_name, value in pipeline_results[name].items():
                        np.testing.assert_almost_equal(value, loaded_[name][objective_name])
                elif name == 'percent_better_than_baseline':
                    np.testing.assert_almost_equal(pipeline_results[name], loaded_[name])
                else:
                    assert pipeline_results[name] == loaded_[name]

    pd.testing.assert_frame_equal(automl.rankings, loaded_automl.rankings)


@patch('cloudpickle.dump')
def test_automl_serialization_protocol(mock_cloudpickle_dump, tmpdir, X_y_binary):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), 'automl.pkl')
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=5, n_jobs=1)

    automl.save(path)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]['protocol'] == cloudpickle.DEFAULT_PROTOCOL

    mock_cloudpickle_dump.reset_mock()
    automl.save(path, pickle_protocol=42)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]['protocol'] == 42


def test_invalid_data_splitter(X_y_binary):
    X, y = X_y_binary
    data_splitter = pd.DataFrame()
    with pytest.raises(ValueError, match='Not a valid data splitter'):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', data_splitter=data_splitter)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
def test_large_dataset_binary(mock_score):
    X = pd.DataFrame({'col_0': [i for i in range(101000)]})
    y = pd.Series([i % 2 for i in range(101000)])

    fraud_objective = FraudCost(amount_col='col_0')

    automl = AutoMLSearch(X_train=X, y_train=y,
                          problem_type='binary',
                          objective=fraud_objective,
                          additional_objectives=['auc', 'f1', 'precision'],
                          max_time=1,
                          max_iterations=1,
                          optimize_thresholds=True,
                          n_jobs=1)
    mock_score.return_value = {automl.objective.name: 1.234}
    automl.search()
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert automl.data_splitter.get_n_splits() == 1

    for pipeline_id in automl.results['search_order']:
        assert len(automl.results['pipeline_results'][pipeline_id]['cv_data']) == 1
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]["mean_cv_score"] == 1.234
        assert automl.results['pipeline_results'][pipeline_id]["mean_cv_score"] == automl.results['pipeline_results'][pipeline_id]['validation_score']


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
def test_large_dataset_multiclass(mock_score):
    X = pd.DataFrame({'col_0': [i for i in range(101000)]})
    y = pd.Series([i % 4 for i in range(101000)])

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', max_time=1, max_iterations=1, n_jobs=1)
    mock_score.return_value = {automl.objective.name: 1.234}
    automl.search()
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert automl.data_splitter.get_n_splits() == 1

    for pipeline_id in automl.results['search_order']:
        assert len(automl.results['pipeline_results'][pipeline_id]['cv_data']) == 1
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]["mean_cv_score"] == 1.234
        assert automl.results['pipeline_results'][pipeline_id]["mean_cv_score"] == automl.results['pipeline_results'][pipeline_id]['validation_score']


@patch('evalml.pipelines.RegressionPipeline.score')
def test_large_dataset_regression(mock_score):
    X = pd.DataFrame({'col_0': [i for i in range(101000)]})
    y = pd.Series([i for i in range(101000)])

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_time=1, max_iterations=1, n_jobs=1)
    mock_score.return_value = {automl.objective.name: 1.234}
    automl.search()
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert automl.data_splitter.get_n_splits() == 1

    for pipeline_id in automl.results['search_order']:
        assert len(automl.results['pipeline_results'][pipeline_id]['cv_data']) == 1
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]["mean_cv_score"] == 1.234
        assert automl.results['pipeline_results'][pipeline_id]["mean_cv_score"] == automl.results['pipeline_results'][pipeline_id]['validation_score']


def test_large_dataset_split_size(X_y_binary):
    X, y = X_y_binary

    def generate_fake_dataset(rows):
        X = pd.DataFrame({'col_0': [i for i in range(rows)]})
        y = pd.Series([i % 2 for i in range(rows)])
        return X, y

    fraud_objective = FraudCost(amount_col='col_0')

    automl = AutoMLSearch(X_train=X, y_train=y,
                          problem_type='binary',
                          objective=fraud_objective,
                          additional_objectives=['auc', 'f1', 'precision'],
                          max_time=1,
                          max_iterations=1,
                          optimize_thresholds=True)
    assert isinstance(automl.data_splitter, StratifiedKFold)

    under_max_rows = _LARGE_DATA_ROW_THRESHOLD - 1
    X, y = generate_fake_dataset(under_max_rows)
    automl = AutoMLSearch(X_train=X, y_train=y,
                          problem_type='binary',
                          objective=fraud_objective,
                          additional_objectives=['auc', 'f1', 'precision'],
                          max_time=1,
                          max_iterations=1,
                          optimize_thresholds=True)
    assert isinstance(automl.data_splitter, StratifiedKFold)

    automl.data_splitter = None
    over_max_rows = _LARGE_DATA_ROW_THRESHOLD + 1
    X, y = generate_fake_dataset(over_max_rows)

    automl = AutoMLSearch(X_train=X, y_train=y,
                          problem_type='binary',
                          objective=fraud_objective,
                          additional_objectives=['auc', 'f1', 'precision'],
                          max_time=1,
                          max_iterations=1,
                          optimize_thresholds=True)
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert automl.data_splitter.test_size == (_LARGE_DATA_PERCENT_VALIDATION)


def test_data_splitter_shuffle():
    # this test checks that the default data split strategy should shuffle data. it creates a target which
    # increases monotonically from 0 to n-1.
    #
    # if shuffle is enabled, the baseline model, which predicts the mean of the training data, should accurately
    # predict the mean of the validation data, because the training split in each CV fold will contain a mix of
    # values from across the target range, thus yielding an R^2 of close to 0.
    #
    # if shuffle is disabled, the mean value learned on each CV fold's training data will be incredible inaccurate,
    # thus yielding an R^2 well below 0.

    n = 100000
    X = pd.DataFrame({'col_0': np.random.random(n)})
    y = pd.Series(np.arange(n), name='target')
    automl = AutoMLSearch(X_train=X, y_train=y,
                          problem_type='regression',
                          max_time=1,
                          max_iterations=1,
                          n_jobs=1)
    automl.search()
    assert automl.results['search_order'] == [0]
    assert len(automl.results['pipeline_results'][0]['cv_data']) == 3
    for fold in range(3):
        np.testing.assert_almost_equal(automl.results['pipeline_results'][0]['cv_data'][fold]["mean_cv_score"], 0.0, decimal=4)
    np.testing.assert_almost_equal(automl.results['pipeline_results'][0]["mean_cv_score"], 0.0, decimal=4)
    np.testing.assert_almost_equal(automl.results['pipeline_results'][0]['validation_score'], 0.0, decimal=4)


def test_allowed_pipelines_with_incorrect_problem_type(dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    # checks that not setting allowed_pipelines does not error out
    AutoMLSearch(X_train=X, y_train=y, problem_type='binary')

    with pytest.raises(ValueError, match="is not compatible with problem_type"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_pipelines=[dummy_binary_pipeline_class({})])


def test_main_objective_problem_type_mismatch(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', objective='R2')
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='regression', objective='MCC Binary')
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', objective='MCC Multiclass')
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', objective='MSE')


def test_init_missing_data(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match=r"Must specify training data as a 2d array using the X_train argument"):
        AutoMLSearch(y_train=y, problem_type='binary')

    with pytest.raises(ValueError, match=r"Must specify training data target values as a 1d vector using the y_train argument"):
        AutoMLSearch(X_train=X, problem_type='binary')


def test_init_problem_type_error(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match=r"choose one of \(binary, multiclass, regression\) as problem_type"):
        AutoMLSearch(X_train=X, y_train=y)

    with pytest.raises(KeyError, match=r"does not exist"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='multi')


def test_init_objective(X_y_binary):
    X, y = X_y_binary
    defaults = {'multiclass': 'Log Loss Multiclass', 'binary': 'Log Loss Binary', 'regression': 'R2'}
    for problem_type in defaults:
        error_automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type)
        assert error_automl.objective.name == defaults[problem_type]


@patch('evalml.automl.automl_search.AutoMLSearch.search')
def test_checks_at_search_time(mock_search, dummy_regression_pipeline_class, X_y_multi):
    X, y = X_y_multi

    error_text = "in search, problem_type mismatches label type."
    mock_search.side_effect = ValueError(error_text)

    error_automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', objective="R2")
    with pytest.raises(ValueError, match=error_text):
        error_automl.search()


def test_incompatible_additional_objectives(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match="is not compatible with a "):
        AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', additional_objectives=['Precision', 'AUC'])


def test_default_objective(X_y_binary):
    X, y = X_y_binary
    correct_matches = {ProblemTypes.MULTICLASS: 'Log Loss Multiclass',
                       ProblemTypes.BINARY: 'Log Loss Binary',
                       ProblemTypes.REGRESSION: 'R2'}
    for problem_type in correct_matches:
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type)
        assert automl.objective.name == correct_matches[problem_type]

        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type.name)
        assert automl.objective.name == correct_matches[problem_type]


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1,
                          allowed_pipelines=[dummy_binary_pipeline_class({})])
    automl.search()
    assert len(automl.rankings) == 1
    assert len(automl.full_rankings) == 1
    original_best_pipeline = automl.best_pipeline
    assert original_best_pipeline is not None

    mock_score.return_value = {'Log Loss Binary': 0.1234}
    test_pipeline = dummy_binary_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline)
    assert automl.best_pipeline.name == test_pipeline.name
    assert automl.best_pipeline.parameters == test_pipeline.parameters
    assert automl.best_pipeline.component_graph == test_pipeline.component_graph
    assert len(automl.rankings) == 2
    assert len(automl.full_rankings) == 2
    assert 0.1234 in automl.rankings["mean_cv_score"].values

    mock_score.return_value = {'Log Loss Binary': 0.5678}
    test_pipeline_2 = dummy_binary_pipeline_class(parameters={'Mock Classifier': {'a': 1.234}})
    automl.add_to_rankings(test_pipeline_2)
    assert automl.best_pipeline.name == test_pipeline.name
    assert automl.best_pipeline.parameters == test_pipeline.parameters
    assert automl.best_pipeline.component_graph == test_pipeline.component_graph
    assert len(automl.rankings) == 2
    assert len(automl.full_rankings) == 3
    assert 0.5678 not in automl.rankings["mean_cv_score"].values
    assert 0.5678 in automl.full_rankings["mean_cv_score"].values


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_no_search(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1,
                          allowed_pipelines=[dummy_binary_pipeline_class({})])

    mock_score.return_value = {'Log Loss Binary': 0.5234}
    test_pipeline = dummy_binary_pipeline_class(parameters={})

    automl.add_to_rankings(test_pipeline)
    best_pipeline = automl.best_pipeline
    assert best_pipeline is not None
    assert isinstance(automl.data_splitter, StratifiedKFold)
    assert len(automl.rankings) == 1
    assert 0.5234 in automl.rankings["mean_cv_score"].values
    assert np.isnan(automl.results['pipeline_results'][0]['percent_better_than_baseline'])
    assert all(np.isnan(res) for res in automl.results['pipeline_results'][0]['percent_better_than_baseline_all_objectives'].values())


@patch('evalml.pipelines.RegressionPipeline.score')
def test_add_to_rankings_regression_large(mock_score, dummy_regression_pipeline_class):
    X = pd.DataFrame({'col_0': [i for i in range(101000)]})
    y = pd.Series([i for i in range(101000)])

    automl = AutoMLSearch(X_train=X, y_train=y, allowed_pipelines=[dummy_regression_pipeline_class({})],
                          problem_type='regression', max_time=1, max_iterations=1, n_jobs=1)
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    mock_score.return_value = {automl.objective.name: 0.1234}

    automl.add_to_rankings(dummy_regression_pipeline_class({}))
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert len(automl.rankings) == 1
    assert 0.1234 in automl.rankings["mean_cv_score"].values


def test_add_to_rankings_new_pipeline(dummy_regression_pipeline_class):
    X = pd.DataFrame({'col_0': [i for i in range(100)]})
    y = pd.Series([i for i in range(100)])

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_time=1, max_iterations=1, n_jobs=1)
    test_pipeline = dummy_regression_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline)


@patch('evalml.pipelines.RegressionPipeline.score')
def test_add_to_rankings_regression(mock_score, dummy_regression_pipeline_class, X_y_regression):
    X, y = X_y_regression

    automl = AutoMLSearch(X_train=X, y_train=y, allowed_pipelines=[dummy_regression_pipeline_class({})],
                          problem_type='regression', max_time=1, max_iterations=1, n_jobs=1)
    mock_score.return_value = {automl.objective.name: 0.1234}

    automl.add_to_rankings(dummy_regression_pipeline_class({}))
    assert isinstance(automl.data_splitter, KFold)
    assert len(automl.rankings) == 1
    assert 0.1234 in automl.rankings["mean_cv_score"].values


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_duplicate(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 0.1234}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1, allowed_pipelines=[dummy_binary_pipeline_class({})])
    automl.search()
    best_pipeline = automl.best_pipeline
    test_pipeline = dummy_binary_pipeline_class(parameters={})
    assert automl.best_pipeline == best_pipeline
    automl.add_to_rankings(test_pipeline)

    test_pipeline_duplicate = dummy_binary_pipeline_class(parameters={})
    assert automl.add_to_rankings(test_pipeline_duplicate) is None


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_trained(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    class CoolBinaryClassificationPipeline(dummy_binary_pipeline_class):
        custom_name = "Cool Binary Classification Pipeline"

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1,
                          allowed_pipelines=[dummy_binary_pipeline_class({}), CoolBinaryClassificationPipeline({})])
    automl.search()
    assert len(automl.rankings) == 1
    assert len(automl.full_rankings) == 1

    mock_score.return_value = {'Log Loss Binary': 0.1234}
    test_pipeline = dummy_binary_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline)
    assert len(automl.rankings) == 2
    assert len(automl.full_rankings) == 2
    assert list(automl.rankings["mean_cv_score"].values).count(0.1234) == 1
    assert list(automl.full_rankings["mean_cv_score"].values).count(0.1234) == 1

    mock_fit.return_value = CoolBinaryClassificationPipeline(parameters={})
    test_pipeline_trained = CoolBinaryClassificationPipeline(parameters={}).fit(X, y)
    automl.add_to_rankings(test_pipeline_trained)
    assert len(automl.rankings) == 3
    assert len(automl.full_rankings) == 3
    assert list(automl.rankings["mean_cv_score"].values).count(0.1234) == 2
    assert list(automl.full_rankings["mean_cv_score"].values).count(0.1234) == 2


def test_no_search(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary')
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)

    df_columns = ["id", "pipeline_name", "mean_cv_score", "standard_deviation_cv_score",
                  "validation_score", "percent_better_than_baseline", "high_variance_cv", "parameters"]
    assert (automl.rankings.columns == df_columns).all()
    assert (automl.full_rankings.columns == df_columns).all()

    with pytest.raises(PipelineNotFoundError):
        automl.best_pipeline

    with pytest.raises(PipelineNotFoundError):
        automl.get_pipeline(0)

    with pytest.raises(PipelineNotFoundError):
        automl.describe_pipeline(0)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_get_pipeline_invalid(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary')
    with pytest.raises(PipelineNotFoundError, match="Pipeline not found in automl results"):
        automl.get_pipeline(1000)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)
    automl.search()
    assert automl.get_pipeline(0).name == 'Mode Baseline Binary Classification Pipeline'
    automl._results['pipeline_results'][0].pop('pipeline_class')
    automl._pipelines_searched.pop(0)

    with pytest.raises(PipelineNotFoundError, match="Pipeline class or parameters not found in automl results"):
        automl.get_pipeline(0)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)
    automl.search()
    assert automl.get_pipeline(0).name == 'Mode Baseline Binary Classification Pipeline'
    automl._results['pipeline_results'][0].pop('parameters')
    with pytest.raises(PipelineNotFoundError, match="Pipeline class or parameters not found in automl results"):
        automl.get_pipeline(0)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_get_pipeline(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)
    automl.search()
    for _, ranking in automl.rankings.iterrows():
        pl = automl.get_pipeline(ranking.id)
        assert pl.parameters == ranking.parameters
        assert pl.name == ranking.pipeline_name
        assert not pl._is_fitted


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={'Log Loss Binary': 1.0})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@pytest.mark.parametrize("return_dict", [True, False])
def test_describe_pipeline(mock_fit, mock_score, return_dict, caplog, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)
    automl.search()
    out = caplog.text

    assert "Searching up to 1 pipelines. " in out

    assert len(automl.results['pipeline_results']) == 1
    caplog.clear()
    automl_dict = automl.describe_pipeline(0, return_dict=return_dict)
    out = caplog.text
    assert "Mode Baseline Binary Classification Pipeline" in out
    assert "Problem Type: binary" in out
    assert "Model Family: Baseline" in out
    assert "* strategy : mode" in out
    assert "Total training time (including CV): " in out
    assert "Log Loss Binary # Training # Validation" in out
    assert "0                      1.000         66           34" in out
    assert "1                      1.000         67           33" in out
    assert "2                      1.000         67           33" in out
    assert "mean                   1.000          -            -" in out
    assert "std                    0.000          -            -" in out
    assert "coef of var            0.000          -            -" in out

    if return_dict:
        assert automl_dict['id'] == 0
        assert automl_dict['pipeline_name'] == 'Mode Baseline Binary Classification Pipeline'
        assert automl_dict['pipeline_summary'] == 'Baseline Classifier'
        assert automl_dict['parameters'] == {'Baseline Classifier': {'strategy': 'mode'}}
        assert automl_dict["mean_cv_score"] == 1.0
        assert not automl_dict['high_variance_cv']
        assert isinstance(automl_dict['training_time'], float)
        assert automl_dict['cv_data'] == [{'all_objective_scores': OrderedDict([('Log Loss Binary', 1.0), ('# Training', 66), ('# Validation', 34)]), "mean_cv_score": 1.0, 'binary_classification_threshold': None},
                                          {'all_objective_scores': OrderedDict([('Log Loss Binary', 1.0), ('# Training', 67), ('# Validation', 33)]), "mean_cv_score": 1.0, 'binary_classification_threshold': None},
                                          {'all_objective_scores': OrderedDict([('Log Loss Binary', 1.0), ('# Training', 67), ('# Validation', 33)]), "mean_cv_score": 1.0, 'binary_classification_threshold': None}]
        assert automl_dict['percent_better_than_baseline_all_objectives'] == {'Log Loss Binary': 0}
        assert automl_dict['percent_better_than_baseline'] == 0
        assert automl_dict['validation_score'] == 1.0
    else:
        assert automl_dict is None


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@pytest.mark.parametrize("return_dict", [True, False])
def test_describe_pipeline_with_ensembling(mock_pipeline_fit, mock_score, return_dict, X_y_binary, caplog):
    X, y = X_y_binary

    two_stacking_batches = 1 + 2 * (len(get_estimators(ProblemTypes.BINARY)) + 1)
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_batches=two_stacking_batches,
                          objective="Log Loss Binary", ensembling=True, error_callback=raise_error_callback)

    mock_score.side_effect = [{'Log Loss Binary': score} for score in np.arange(0, -1 * automl.max_iterations * automl.data_splitter.get_n_splits(), -0.1)]  # Dcreases with each call
    automl.search()
    pipeline_names = automl.rankings['pipeline_name']
    assert pipeline_names.str.contains('Ensemble').any()

    ensemble_ids = [_get_first_stacked_classifier_no() - 1, len(automl.results['pipeline_results']) - 1]

    for i, ensemble_id in enumerate(ensemble_ids):
        caplog.clear()
        automl_dict = automl.describe_pipeline(ensemble_id, return_dict=return_dict)
        out = caplog.text
        assert "Stacked Ensemble Classification Pipeline" in out
        assert "Problem Type: binary" in out
        assert "Model Family: Ensemble" in out
        assert "* final_estimator : None" in out
        assert "Total training time (including CV): " in out
        assert "Log Loss Binary # Training # Validation" in out
        assert "Input for ensembler are pipelines with IDs:" in out

        if return_dict:
            assert automl_dict['id'] == ensemble_id
            assert automl_dict['pipeline_name'] == "Stacked Ensemble Classification Pipeline"
            assert automl_dict['pipeline_summary'] == 'Stacked Ensemble Classifier'
            assert isinstance(automl_dict["mean_cv_score"], float)
            assert not automl_dict['high_variance_cv']
            assert isinstance(automl_dict['training_time'], float)
            assert isinstance(automl_dict['percent_better_than_baseline_all_objectives'], dict)
            assert isinstance(automl_dict['percent_better_than_baseline'], float)
            assert isinstance(automl_dict['validation_score'], float)
            assert len(automl_dict['input_pipeline_ids']) == len(allowed_model_families("binary"))
            if i == 0:
                assert all(input_id < ensemble_id for input_id in automl_dict['input_pipeline_ids'])
            else:
                assert all(input_id < ensemble_id for input_id in automl_dict['input_pipeline_ids'])
                assert all(input_id > ensemble_ids[0] for input_id in automl_dict['input_pipeline_ids'])
        else:
            assert automl_dict is None


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_results_getter(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)

    assert automl.results == {'pipeline_results': {},
                              'search_order': []}

    mock_score.return_value = {'Log Loss Binary': 1.0}
    automl.search()

    assert automl.results['pipeline_results'][0]["mean_cv_score"] == 1.0

    with pytest.raises(AttributeError, match='set attribute'):
        automl.results = 2.0

    automl.results['pipeline_results'][0]["mean_cv_score"] = 2.0
    assert automl.results['pipeline_results'][0]["mean_cv_score"] == 1.0


@pytest.mark.parametrize("data_type", ['li', 'np', 'pd', 'ww'])
@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@pytest.mark.parametrize("target_type", ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool', 'category', 'object', 'Int64', 'boolean'])
def test_targets_pandas_data_types_classification(data_type, automl_type, target_type, make_data_type):
    if data_type == 'np' and target_type in ['Int64', 'boolean']:
        pytest.skip("Skipping test where data type is numpy and target type is nullable dtype")

    if automl_type == ProblemTypes.BINARY:
        X, y = load_breast_cancer(return_pandas=True)
        if "bool" in target_type:
            y = y.map({"malignant": False, "benign": True})
    elif automl_type == ProblemTypes.MULTICLASS:
        if "bool" in target_type:
            pytest.skip("Skipping test where problem type is multiclass but target type is boolean")
        X, y = load_wine(return_pandas=True)
    unique_vals = y.unique()
    # Update target types as necessary
    if target_type in ['category', 'object']:
        if target_type == "category":
            y = pd.Categorical(y)
    elif "int" in target_type.lower():
        y = y.map({unique_vals[i]: int(i) for i in range(len(unique_vals))})
    elif "float" in target_type.lower():
        y = y.map({unique_vals[i]: float(i) for i in range(len(unique_vals))})

    y = y.astype(target_type)
    if data_type != 'pd':
        X = make_data_type(data_type, X)
        y = make_data_type(data_type, y)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, max_iterations=3, n_jobs=1)
    automl.search()
    for pipeline_id, pipeline_result in automl.results['pipeline_results'].items():
        cv_data = pipeline_result['cv_data']
        for fold in cv_data:
            all_objective_scores = fold["all_objective_scores"]
            for score in all_objective_scores.values():
                assert score is not None

    assert len(automl.full_rankings) == 3
    assert not automl.full_rankings["mean_cv_score"].isnull().values.any()


class KeyboardInterruptOnKthPipeline:
    """Helps us time when the test will send a KeyboardInterrupt Exception to search."""

    def __init__(self, k, starting_index):
        self.n_calls = starting_index
        self.k = k

    def __call__(self):
        """Raises KeyboardInterrupt on the kth call.
        Arguments are ignored but included to meet the call back API.
        """
        if self.n_calls == self.k:
            self.n_calls += 1
            raise KeyboardInterrupt
        else:
            self.n_calls += 1
            return True


# These are used to mock return values to the builtin "input" function.
interrupt = ["y"]
interrupt_after_bad_message = ["No.", "Yes!", "y"]
dont_interrupt = ["n"]
dont_interrupt_after_bad_message = ["Yes", "yes.", "n"]


@pytest.mark.parametrize("when_to_interrupt,user_input,number_results",
                         [(1, interrupt, 0),
                          (1, interrupt_after_bad_message, 0)])
@patch("builtins.input")
@patch('evalml.automl.engine.sequential_engine.SequentialComputation.get_result')
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"F1": 1.0})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_catch_keyboard_interrupt_baseline(mock_fit, mock_score, mock_future_get_result, mock_input,
                                           when_to_interrupt, user_input, number_results,
                                           X_y_binary):
    X, y = X_y_binary

    mock_input.side_effect = user_input
    mock_future_get_result.side_effect = KeyboardInterruptOnKthPipeline(k=when_to_interrupt, starting_index=1)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=5,
                          objective="f1")
    automl.search()
    assert len(automl._results['pipeline_results']) == number_results
    if number_results == 0:
        with pytest.raises(PipelineNotFoundError):
            _ = automl.best_pipeline


@pytest.mark.parametrize("when_to_interrupt,user_input,number_results",
                         [(1, dont_interrupt, 5),
                          (1, dont_interrupt_after_bad_message, 5),
                          (2, interrupt, 1),
                          (2, interrupt_after_bad_message, 1),
                          (2, dont_interrupt, 5),
                          (2, dont_interrupt_after_bad_message, 5),
                          (3, interrupt, 2),
                          (3, interrupt_after_bad_message, 2),
                          (3, dont_interrupt, 5),
                          (3, dont_interrupt_after_bad_message, 5),
                          (5, interrupt, 4),
                          (5, interrupt_after_bad_message, 4),
                          (5, dont_interrupt, 5),
                          (5, dont_interrupt_after_bad_message, 5)])
@patch("builtins.input")
@patch('evalml.automl.engine.sequential_engine.SequentialComputation.done')
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"F1": 1.0})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_catch_keyboard_interrupt(mock_fit, mock_score, mock_future_get_result, mock_input,
                                  when_to_interrupt, user_input, number_results,
                                  X_y_binary):
    X, y = X_y_binary

    mock_input.side_effect = user_input
    mock_future_get_result.side_effect = KeyboardInterruptOnKthPipeline(k=when_to_interrupt, starting_index=2)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=5,
                          objective="f1", optimize_thresholds=False)
    automl.search()
    assert len(automl._results['pipeline_results']) == number_results


@patch("builtins.input", return_value="Y")
@patch('evalml.automl.engine.sequential_engine.SequentialComputation.done',
       side_effect=KeyboardInterruptOnKthPipeline(k=4, starting_index=2))
@patch('evalml.automl.engine.sequential_engine.SequentialComputation.cancel')
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"F1": 1.0})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_jobs_cancelled_when_keyboard_interrupt(mock_fit, mock_score, mock_cancel, mock_done, mock_input, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=6,
                          objective="f1", optimize_thresholds=False)
    automl.search()
    assert len(automl._results['pipeline_results']) == 3

    # Since we trigger KeyBoardInterrupt the 4th time we call done, we've successfully evaluated the baseline plus 2
    # pipelines in the first batch. Since there are len(automl.allowed_pipelines) pipelines in the first batch,
    # we should cancel len(automl.allowed_pipelines) - 2 computations
    assert mock_cancel.call_count == len(automl.allowed_pipelines) - 3 + 1


def make_mock_rankings(scores):
    df = pd.DataFrame({'id': range(len(scores)), "mean_cv_score": scores,
                       'pipeline_name': [f'Mock name {i}' for i in range(len(scores))]})
    return df


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.automl.AutoMLSearch.full_rankings', new_callable=PropertyMock)
@patch('evalml.automl.AutoMLSearch.rankings', new_callable=PropertyMock)
def test_pipelines_in_batch_return_nan(mock_rankings, mock_full_rankings, mock_next_batch, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    mock_rankings.side_effect = [make_mock_rankings([0, 0, 0]),  # first batch
                                 make_mock_rankings([0, 0, 0, 0, np.nan]),  # second batch
                                 make_mock_rankings([0, 0, 0, 0, np.nan, np.nan, np.nan])]  # third batch, should raise error
    mock_full_rankings.side_effect = [make_mock_rankings([0, 0, 0]),  # first batch
                                      make_mock_rankings([0, 0, 0, 0, np.nan]),  # second batch
                                      make_mock_rankings([0, 0, 0, 0, np.nan, np.nan, np.nan])]  # third batch, should raise error
    mock_next_batch.side_effect = [[dummy_binary_pipeline_class(parameters={}), dummy_binary_pipeline_class(parameters={})] for i in range(3)]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=3, allowed_pipelines=[dummy_binary_pipeline_class({})], n_jobs=1)
    with pytest.raises(AutoMLSearchException, match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"):
        automl.search()


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.automl.AutoMLSearch.full_rankings', new_callable=PropertyMock)
@patch('evalml.automl.AutoMLSearch.rankings', new_callable=PropertyMock)
def test_pipelines_in_batch_return_none(mock_rankings, mock_full_rankings, mock_next_batch, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    mock_rankings.side_effect = [make_mock_rankings([0, 0, 0]),  # first batch
                                 make_mock_rankings([0, 0, 0, 0, None]),  # second batch
                                 make_mock_rankings([0, 0, 0, 0, None, None, None])]  # third batch, should raise error
    mock_full_rankings.side_effect = [make_mock_rankings([0, 0, 0]),  # first batch
                                      make_mock_rankings([0, 0, 0, 0, None]),  # second batch
                                      make_mock_rankings([0, 0, 0, 0, None, None, None])]  # third batch, should raise error
    mock_next_batch.side_effect = [[dummy_binary_pipeline_class(parameters={}), dummy_binary_pipeline_class(parameters={})] for i in range(3)]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=3, allowed_pipelines=[dummy_binary_pipeline_class({})], n_jobs=1)
    with pytest.raises(AutoMLSearchException, match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"):
        automl.search()


@patch('evalml.automl.engine.engine_base.split_data')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_error_during_train_test_split(mock_fit, mock_score, mock_split_data, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}
    # this method is called during pipeline eval for binary classification and will cause scores to be set to nan
    mock_split_data.side_effect = RuntimeError()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', objective='Accuracy Binary', max_iterations=2, optimize_thresholds=True, train_best_pipeline=False)
    with pytest.raises(AutoMLSearchException, match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"):
        automl.search()
    for pipeline in automl.results['pipeline_results'].values():
        assert np.isnan(pipeline["mean_cv_score"])


all_objectives = get_core_objectives("binary") + get_core_objectives("multiclass") + get_core_objectives("regression")


class CustomClassificationObjective(BinaryClassificationObjective):
    """Accuracy score for binary and multiclass classification."""
    name = "Classification Accuracy"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def objective_function(self, y_true, y_predicted, X=None):
        """Not implementing since mocked in our tests."""


class CustomRegressionObjective(RegressionObjective):
    """Accuracy score for binary and multiclass classification."""
    name = "Custom Regression Objective"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False
    problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def objective_function(self, y_true, y_predicted, X=None):
        """Not implementing since mocked in our tests."""


@pytest.mark.parametrize("objective,pipeline_scores,baseline_score,problem_type_value",
                         product(all_objectives + [CostBenefitMatrix, CustomClassificationObjective()],
                                 [(0.3, 0.4), (np.nan, 0.4), (0.3, np.nan), (np.nan, np.nan)],
                                 [0.1, np.nan],
                                 [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]))
def test_percent_better_than_baseline_in_rankings(objective, pipeline_scores, baseline_score, problem_type_value,
                                                  dummy_binary_pipeline_class, dummy_multiclass_pipeline_class,
                                                  dummy_regression_pipeline_class,
                                                  dummy_time_series_regression_pipeline_class,
                                                  X_y_binary):
    if not objective.is_defined_for_problem_type(problem_type_value):
        pytest.skip("Skipping because objective is not defined for problem type")

    # Ok to only use binary labels since score and fit methods are mocked
    X, y = X_y_binary

    pipeline_class = {ProblemTypes.BINARY: dummy_binary_pipeline_class,
                      ProblemTypes.MULTICLASS: dummy_multiclass_pipeline_class,
                      ProblemTypes.REGRESSION: dummy_regression_pipeline_class,
                      ProblemTypes.TIME_SERIES_REGRESSION: dummy_time_series_regression_pipeline_class}[problem_type_value]
    baseline_pipeline_class = {ProblemTypes.BINARY: "evalml.pipelines.BinaryClassificationPipeline",
                               ProblemTypes.MULTICLASS: "evalml.pipelines.MulticlassClassificationPipeline",
                               ProblemTypes.REGRESSION: "evalml.pipelines.RegressionPipeline",
                               ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesRegressionPipeline"
                               }[problem_type_value]

    class DummyPipeline(pipeline_class):
        problem_type = problem_type_value

        def __init__(self, parameters, random_seed=0):
            super().__init__(parameters=parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

        def fit(self, *args, **kwargs):
            """Mocking fit"""

    class Pipeline1(DummyPipeline):
        custom_name = "Pipeline1"

    class Pipeline2(DummyPipeline):
        custom_name = "Pipeline2"

    mock_score_1 = MagicMock(return_value={objective.name: pipeline_scores[0]})
    mock_score_2 = MagicMock(return_value={objective.name: pipeline_scores[1]})
    Pipeline1.score = mock_score_1
    Pipeline2.score = mock_score_2

    if objective.name.lower() == "cost benefit matrix":
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type_value, max_iterations=3,
                              allowed_pipelines=[Pipeline1({}), Pipeline2({})], objective=objective(0, 0, 0, 0),
                              additional_objectives=[], optimize_thresholds=False, n_jobs=1)
    elif problem_type_value == ProblemTypes.TIME_SERIES_REGRESSION:
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type_value, max_iterations=3,
                              allowed_pipelines=[Pipeline1({'pipeline': {'date_index': None, 'gap': 0, 'max_delay': 0}}), Pipeline2({'pipeline': {'date_index': None, 'gap': 0, 'max_delay': 0}})], objective=objective,
                              additional_objectives=[], problem_configuration={'date_index': None, 'gap': 0, 'max_delay': 0}, train_best_pipeline=False, n_jobs=1)
    else:
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type_value, max_iterations=3,
                              allowed_pipelines=[Pipeline1({}), Pipeline2({})], objective=objective,
                              additional_objectives=[], optimize_thresholds=False, n_jobs=1)

    with patch(baseline_pipeline_class + ".score", return_value={objective.name: baseline_score}):
        if np.isnan(pipeline_scores).all():
            with pytest.raises(AutoMLSearchException, match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"):
                automl.search()
        else:
            automl.search()
        scores = dict(zip(automl.rankings.pipeline_name, automl.rankings.percent_better_than_baseline))
        baseline_name = next(name for name in automl.rankings.pipeline_name if name not in {"Pipeline1", "Pipeline2"})
        answers = {"Pipeline1": round(objective.calculate_percent_difference(pipeline_scores[0], baseline_score), 2),
                   "Pipeline2": round(objective.calculate_percent_difference(pipeline_scores[1], baseline_score), 2),
                   baseline_name: round(objective.calculate_percent_difference(baseline_score, baseline_score), 2)}
        for name in answers:
            np.testing.assert_almost_equal(scores[name], answers[name], decimal=3)


@pytest.mark.parametrize("custom_additional_objective", [True, False])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression", "time series regression"])
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.MulticlassClassificationPipeline.fit")
@patch("evalml.pipelines.RegressionPipeline.fit")
@patch("evalml.pipelines.TimeSeriesRegressionPipeline.fit")
def test_percent_better_than_baseline_computed_for_all_objectives(mock_time_series_baseline_regression_fit,
                                                                  mock_regression_fit,
                                                                  mock_multiclass_fit,
                                                                  mock_binary_fit,
                                                                  problem_type,
                                                                  custom_additional_objective,
                                                                  dummy_binary_pipeline_class,
                                                                  dummy_multiclass_pipeline_class,
                                                                  dummy_regression_pipeline_class,
                                                                  dummy_time_series_regression_pipeline_class,
                                                                  X_y_binary):
    X, y = X_y_binary

    problem_type_enum = handle_problem_types(problem_type)

    pipeline_class = {"binary": dummy_binary_pipeline_class,
                      "multiclass": dummy_multiclass_pipeline_class,
                      "regression": dummy_regression_pipeline_class,
                      "time series regression": dummy_time_series_regression_pipeline_class}[problem_type]
    baseline_pipeline_class = {ProblemTypes.BINARY: "evalml.pipelines.BinaryClassificationPipeline",
                               ProblemTypes.MULTICLASS: "evalml.pipelines.MulticlassClassificationPipeline",
                               ProblemTypes.REGRESSION: "evalml.pipelines.RegressionPipeline",
                               ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesRegressionPipeline"
                               }[problem_type_enum]

    class DummyPipeline(pipeline_class):
        name = "Dummy 1"
        problem_type = problem_type_enum

        def __init__(self, parameters, random_seed=0):
            super().__init__(parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

        def fit(self, *args, **kwargs):
            """Mocking fit"""

    additional_objectives = None
    if custom_additional_objective:
        if CustomClassificationObjective.is_defined_for_problem_type(problem_type_enum):
            additional_objectives = [CustomClassificationObjective()]
        else:
            additional_objectives = [CustomRegressionObjective(), "Root Mean Squared Error"]

    core_objectives = get_core_objectives(problem_type)
    if additional_objectives:
        core_objectives = [get_default_primary_search_objective(problem_type_enum)] + additional_objectives
    mock_scores = {get_objective(obj).name: i for i, obj in enumerate(core_objectives)}
    mock_baseline_scores = {get_objective(obj).name: i + 1 for i, obj in enumerate(core_objectives)}
    answer = {}
    baseline_percent_difference = {}
    for obj in core_objectives:
        obj_class = get_objective(obj)
        answer[obj_class.name] = obj_class.calculate_percent_difference(mock_scores[obj_class.name],
                                                                        mock_baseline_scores[obj_class.name])
        baseline_percent_difference[obj_class.name] = 0

    mock_score_1 = MagicMock(return_value=mock_scores)
    DummyPipeline.score = mock_score_1
    parameters = {}
    if problem_type_enum == ProblemTypes.TIME_SERIES_REGRESSION:
        parameters = {"pipeline": {'date_index': None, "gap": 6, "max_delay": 3}}
    # specifying problem_configuration for all problem types for conciseness
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type, max_iterations=2,
                          allowed_pipelines=[DummyPipeline(parameters)],
                          objective="auto", problem_configuration={'date_index': None, 'gap': 1, 'max_delay': 1},
                          additional_objectives=additional_objectives)

    with patch(baseline_pipeline_class + ".score", return_value=mock_baseline_scores):
        automl.search()
        assert len(automl.results['pipeline_results']) == 2, "This tests assumes only one non-baseline pipeline was run!"
        pipeline_results = automl.results['pipeline_results'][1]
        baseline_results = automl.results['pipeline_results'][0]
        assert pipeline_results["percent_better_than_baseline_all_objectives"] == answer
        assert pipeline_results['percent_better_than_baseline'] == pipeline_results["percent_better_than_baseline_all_objectives"][automl.objective.name]
        # Check that baseline is 0% better than baseline
        assert baseline_results["percent_better_than_baseline_all_objectives"] == baseline_percent_difference


@pytest.mark.parametrize("fold_scores", [[2, 4, 6], [np.nan, 4, 6]])
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={'Log Loss Binary': 1, 'F1': 1})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_percent_better_than_baseline_scores_different_folds(mock_fit,
                                                             mock_score,
                                                             fold_scores,
                                                             dummy_binary_pipeline_class,
                                                             X_y_binary):
    # Test that percent-better-than-baseline is correctly computed when scores differ across folds
    X, y = X_y_binary

    class DummyPipeline(dummy_binary_pipeline_class):
        name = "Dummy 1"
        problem_type = ProblemTypes.BINARY

        def __init__(self, parameters, random_seed=0):
            super().__init__(parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    mock_score = MagicMock(side_effect=[{"Log Loss Binary": 1, "F1": val} for val in fold_scores])
    DummyPipeline.score = mock_score
    f1 = get_objective("f1")()

    if np.isnan(fold_scores[0]):
        answer = np.nan
    else:
        answer = f1.calculate_percent_difference(4, 1)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=2,
                          allowed_pipelines=[DummyPipeline({})], objective="log loss binary", additional_objectives=["f1"])

    automl.search()
    assert len(automl.results['pipeline_results']) == 2, "This tests assumes only one non-baseline pipeline was run!"
    pipeline_results = automl.results['pipeline_results'][1]
    np.testing.assert_equal(pipeline_results["percent_better_than_baseline_all_objectives"]['F1'], answer)


def _get_first_stacked_classifier_no(model_families=None):
    """Gets the number of iterations necessary before the stacked ensemble will be used."""
    num_classifiers = len(get_estimators(ProblemTypes.BINARY, model_families=model_families))
    # Baseline + first batch + each pipeline iteration (5 is current default pipelines_per_batch) + 1
    return 1 + num_classifiers + num_classifiers * 5 + 1


@pytest.mark.parametrize("max_iterations", [None, 1, 8, 10, _get_first_stacked_classifier_no(), _get_first_stacked_classifier_no() + 2])
@pytest.mark.parametrize("use_ensembling", [True, False])
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_max_iteration_works_with_stacked_ensemble(mock_pipeline_fit, mock_score, max_iterations, use_ensembling, X_y_binary, caplog):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations, objective="Log Loss Binary", ensembling=use_ensembling)
    automl.search()
    # every nth batch a stacked ensemble will be trained
    if max_iterations is None:
        max_iterations = 5  # Default value for max_iterations

    pipeline_names = automl.rankings['pipeline_name']
    if max_iterations < _get_first_stacked_classifier_no():
        assert not pipeline_names.str.contains('Ensemble').any()
        assert not automl.ensembling_indices
    elif use_ensembling:
        assert pipeline_names.str.contains('Ensemble').any()
        assert f"Ensembling will run at the {_get_first_stacked_classifier_no()} iteration" in caplog.text
        assert automl.ensembling_indices

    else:
        assert not pipeline_names.str.contains('Ensemble').any()
        assert not automl.ensembling_indices


@pytest.mark.parametrize("max_batches", [None, 1, 5, 8, 9, 10, 12, 20])
@pytest.mark.parametrize("use_ensembling", [True, False])
@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.REGRESSION])
@patch('evalml.pipelines.RegressionPipeline.score', return_value={"R2": 0.8})
@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_max_batches_works(mock_pipeline_fit, mock_score, mock_regression_fit, mock_regression_score,
                           max_batches, use_ensembling, problem_type, X_y_binary, X_y_regression):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=None,
                              max_batches=max_batches, ensembling=use_ensembling)
    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type="regression", max_iterations=None,
                              max_batches=max_batches, ensembling=use_ensembling)

    automl.search()
    # every nth batch a stacked ensemble will be trained
    ensemble_nth_batch = len(automl.allowed_pipelines) + 1

    if max_batches is None:
        n_results = len(automl.allowed_pipelines) + 1
        max_batches = 1
        # _automl_algorithm will include all allowed_pipelines in the first batch even
        # if they are not searched over. That is why n_automl_pipelines does not equal
        # n_results when max_iterations and max_batches are None
        n_automl_pipelines = 1 + len(automl.allowed_pipelines)
        num_ensemble_batches = 0
    else:
        # automl algorithm does not know about the additional stacked ensemble pipelines
        num_ensemble_batches = (max_batches - 1) // ensemble_nth_batch if use_ensembling else 0
        # So that the test does not break when new estimator classes are added
        n_results = 1 + len(automl.allowed_pipelines) + (5 * (max_batches - 1 - num_ensemble_batches)) + num_ensemble_batches
        n_automl_pipelines = n_results
    assert automl._automl_algorithm.batch_number == max_batches
    assert automl._automl_algorithm.pipeline_number + 1 == n_automl_pipelines
    assert len(automl.results["pipeline_results"]) == n_results
    if num_ensemble_batches == 0:
        assert automl.rankings.shape[0] == min(1 + len(automl.allowed_pipelines), n_results)  # add one for baseline
    else:
        assert automl.rankings.shape[0] == min(2 + len(automl.allowed_pipelines), n_results)  # add two for baseline and stacked ensemble
    assert automl.full_rankings.shape[0] == n_results


def test_early_stopping_negative(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match='patience value must be a positive integer.'):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', objective='AUC', max_iterations=5, allowed_model_families=['linear_model'], patience=-1, random_seed=0)
    with pytest.raises(ValueError, match='tolerance value must be'):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', objective='AUC', max_iterations=5, allowed_model_families=['linear_model'], patience=1, tolerance=1.5, random_seed=0)


def test_early_stopping(caplog, logistic_regression_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', objective='AUC', max_iterations=5,
                          allowed_model_families=['linear_model'], patience=2, tolerance=0.05,
                          random_seed=0, n_jobs=1)
    mock_results = {
        'search_order': [0, 1, 2, 3],
        'pipeline_results': {}
    }

    scores = [0.84, 0.95, 0.84, 0.96]  # 0.96 is only 1% greater so it doesn't trigger patience due to tolerance
    for id in mock_results['search_order']:
        mock_results['pipeline_results'][id] = {}
        mock_results['pipeline_results'][id]["mean_cv_score"] = scores[id]
        mock_results['pipeline_results'][id]['pipeline_class'] = logistic_regression_binary_pipeline_class
    automl._results = mock_results

    assert not automl._should_continue()
    out = caplog.text
    assert "2 iterations without improvement. Stopping search early." in out


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_one_allowed_pipeline_ensembling_disabled(mock_pipeline_fit, mock_score, X_y_binary, logistic_regression_binary_pipeline_class, caplog):
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.RANDOM_FOREST]) + 1
    # Checks that when len(allowed_pipeline) == 1, ensembling is not run, even if set to True
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations, allowed_model_families=[ModelFamily.RANDOM_FOREST], ensembling=True)
    automl.search()
    assert "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run." in caplog.text

    pipeline_names = automl.rankings['pipeline_name']
    assert not pipeline_names.str.contains('Ensemble').any()

    caplog.clear()
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.LINEAR_MODEL]) + 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations, allowed_pipelines=[logistic_regression_binary_pipeline_class({})], ensembling=True)
    automl.search()
    pipeline_names = automl.rankings['pipeline_name']
    assert not pipeline_names.str.contains('Ensemble').any()
    assert "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run." in caplog.text
    assert not automl.ensembling_indices
    # Check that ensembling runs when len(allowed_model_families) == 1 but len(allowed_pipelines) > 1
    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations, allowed_model_families=[ModelFamily.LINEAR_MODEL], ensembling=True)
    automl.search()
    pipeline_names = automl.rankings['pipeline_name']
    assert pipeline_names.str.contains('Ensemble').any()
    assert "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run." not in caplog.text
    assert automl.ensembling_indices


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_max_iterations_less_than_ensembling_disabled(mock_pipeline_fit, mock_score, X_y_binary, caplog):
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.LINEAR_MODEL])
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations - 1, allowed_model_families=[ModelFamily.LINEAR_MODEL], ensembling=True)
    automl.search()
    assert f"Ensembling is set to True, but max_iterations is too small, so ensembling will not run. Set max_iterations >= {max_iterations} to run ensembling." in caplog.text

    pipeline_names = automl.rankings['pipeline_name']
    assert not pipeline_names.str.contains('Ensemble').any()
    assert not automl.ensembling_indices


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_max_batches_less_than_ensembling_disabled(mock_pipeline_fit, mock_score, X_y_binary, caplog):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_batches=2, allowed_model_families=[ModelFamily.LINEAR_MODEL], ensembling=True)
    automl.search()
    first_ensemble_batch = 1 + len(automl.allowed_pipelines) + 1  # First batch + each pipeline batch
    assert f"Ensembling is set to True, but max_batches is too small, so ensembling will not run. Set max_batches >= {first_ensemble_batch} to run ensembling." in caplog.text

    pipeline_names = automl.rankings['pipeline_name']
    assert not pipeline_names.str.contains('Ensemble').any()
    assert not automl.ensembling_indices


@pytest.mark.parametrize("max_batches", [1, 2, 5, 10])
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_max_batches_output(mock_pipeline_fit, mock_score, max_batches, X_y_binary, caplog):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=None, max_batches=max_batches)
    automl.search()

    output = caplog.text
    assert output.count("Batch Number") == max_batches


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_max_batches_plays_nice_with_other_stopping_criteria(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary

    # Use the old default when all are None
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", objective="Log Loss Binary")
    automl.search()
    assert len(automl.results["pipeline_results"]) == len(get_estimators(problem_type='binary')) + 1

    # Use max_iterations when both max_iterations and max_batches are set
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", objective="Log Loss Binary", max_batches=10,
                          max_iterations=6)
    automl.search()
    assert len(automl.results["pipeline_results"]) == 6

    # Don't change max_iterations when only max_iterations is set
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=4)
    automl.search()
    assert len(automl.results["pipeline_results"]) == 4


@pytest.mark.parametrize("max_batches", [-1, -10, -np.inf])
def test_max_batches_must_be_non_negative(max_batches, X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match=f"Parameter max_batches must be None or non-negative. Received {max_batches}."):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_batches=max_batches)


def test_stopping_criterion_bad(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(TypeError, match=r"Parameter max_time must be a float, int, string or None. Received <class 'tuple'> with value \('test',\)."):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_time=('test',))
    with pytest.raises(ValueError, match=f"Parameter max_batches must be None or non-negative. Received -1."):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_batches=-1)
    with pytest.raises(ValueError, match=f"Parameter max_time must be None or non-negative. Received -1."):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_time=-1)
    with pytest.raises(ValueError, match=f"Parameter max_iterations must be None or non-negative. Received -1."):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=-1)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_data_splitter_binary(mock_fit, mock_score, X_y_binary):
    mock_score.return_value = {'Log Loss Binary': 1.0}
    X, y = X_y_binary
    y[:] = 0
    y[0] = 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", n_jobs=1)
    with pytest.raises(Exception, match="Missing target values in the"):
        with pytest.warns(UserWarning):
            automl.search()

    y[1] = 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", n_jobs=1)
    with pytest.raises(Exception, match="Missing target values in the"):
        with pytest.warns(UserWarning):
            automl.search()

    y[2] = 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", n_jobs=1)
    automl.search()


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
def test_data_splitter_multi(mock_fit, mock_score, X_y_multi):
    mock_score.return_value = {'Log Loss Multiclass': 1.0}
    X, y = X_y_multi
    y[:] = 1
    y[0] = 0

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            automl.search()

    y[1] = 2
    # match based on regex, since data split doesn't have a random seed for reproducibility
    # regex matches the set {} and expects either 2 sets (missing in both train and test)
    #   or 1 set of multiple elements (both missing in train or both in test)
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    with pytest.raises(Exception, match=r"(\{\d?\}.+\{\d?\})|(\{.+\,.+\})"):
        with pytest.warns(UserWarning):
            automl.search()

    y[1] = 0
    y[2:4] = 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            automl.search()

    y[4] = 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            automl.search()

    y[5] = 0
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    automl.search()


@patch('evalml.tuners.skopt_tuner.SKOptTuner.add')
def test_iterative_algorithm_pipeline_hyperparameters_make_pipeline_other_errors(mock_add, X_y_multi):
    X, y = X_y_multi
    custom_hyperparameters = {
        "Imputer": {
            "numeric_impute_strategy": ["most_frequent", "mean"]
        }
    }
    estimators = get_estimators('multiclass', [ModelFamily.EXTRA_TREES])

    pipelines = [make_pipeline(X, y, estimator, 'multiclass', None, custom_hyperparameters) for estimator in estimators]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', allowed_pipelines=pipelines, n_jobs=1)

    mock_add.side_effect = ValueError("Alternate error that can be thrown")
    with pytest.raises(ValueError) as error:
        automl.search()
    assert "Alternate error that can be thrown" in str(error.value)
    assert "Default parameters for components" not in str(error.value)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_iterative_algorithm_pipeline_hyperparameters_make_pipeline_errors(mock_fit, mock_score, X_y_multi):
    X, y = X_y_multi
    invalid_custom_hyperparameters = {
        "Imputer": {
            "numeric_impute_strategy": ["most_frequent", "median"]
        }
    }
    larger_invalid = {
        "Imputer": {
            "numeric_impute_strategy": ["most_frequent", "mean"]
        },
        "Extra Trees Classifier": {
            "max_depth": [4, 5, 6, 7],
            "max_features": ["sqrt", "log2"]
        }
    }
    estimators = get_estimators('multiclass', [ModelFamily.EXTRA_TREES])

    invalid_pipelines = [make_pipeline(X, y, estimator, 'multiclass', None, invalid_custom_hyperparameters) for estimator in estimators]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', allowed_pipelines=invalid_pipelines)
    with pytest.raises(ValueError, match="Default parameters for components"):
        automl.search()

    invalid_pipelines = [make_pipeline(X, y, estimator, 'multiclass', None, larger_invalid) for estimator in estimators]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', allowed_pipelines=invalid_pipelines)
    with pytest.raises(ValueError, match="Default parameters for components"):
        automl.search()


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_iterative_algorithm_pipeline_hyperparameters_make_pipeline(mock_fit, mock_score, X_y_multi):
    X, y = X_y_multi
    custom_hyperparameters = {
        "Imputer": {
            "numeric_impute_strategy": ["mean"]
        }
    }
    larger_custom = {
        "Imputer": {
            "numeric_impute_strategy": ["most_frequent", "mean"]
        },
        "Extra Trees Classifier": {
            "max_depth": [4, 5, 6, 7],
            "max_features": ["auto", "log2"]
        }
    }
    estimators = get_estimators('multiclass', [ModelFamily.EXTRA_TREES])
    pipelines = [make_pipeline(X, y, estimator, 'multiclass', None, custom_hyperparameters) for estimator in estimators]

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', allowed_pipelines=pipelines)
    automl.search()
    assert automl.best_pipeline.hyperparameters['Imputer']['numeric_impute_strategy'] == ["mean"]

    invalid_pipelines = [make_pipeline(X, y, estimator, 'multiclass', None, larger_custom) for estimator in estimators]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', allowed_pipelines=invalid_pipelines)
    automl.search()

    assert automl.best_pipeline.hyperparameters['Imputer']['numeric_impute_strategy'] == ["most_frequent", "mean"]


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.6})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_iterative_algorithm_passes_njobs_to_pipelines(mock_fit, mock_score, dummy_binary_pipeline_class,
                                                       X_y_binary):
    X, y = X_y_binary

    class MockEstimatorWithNJobs(Estimator):
        name = "Mock Classifier with njobs"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        hyperparameter_ranges = {}

        def __init__(self, n_jobs=-1, random_seed=0):
            super().__init__(parameters={"n_jobs": n_jobs}, component_obj=None, random_seed=random_seed)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', n_jobs=3, max_batches=2,
                          allowed_pipelines=[BinaryClassificationPipeline([MockEstimatorWithNJobs], custom_name="Pipeline 1"),
                                             BinaryClassificationPipeline([MockEstimatorWithNJobs], custom_name="Pipeline 2"),
                                             dummy_binary_pipeline_class({})])
    automl.search()
    for parameters in automl.full_rankings.parameters:
        if "Mock Classifier with njobs" in parameters:
            assert parameters["Mock Classifier with njobs"]["n_jobs"] == 3
        else:
            assert all("n_jobs" not in component_params for component_params in parameters.values())


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_ensembling_false(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time='60 seconds', max_batches=20, ensembling=False)
    automl.search()
    assert not automl.rankings['pipeline_name'].str.contains('Ensemble').any()


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_search_with_text(mock_fit, mock_score):
    X = pd.DataFrame(
        {'col_1': ['I\'m singing in the rain! Just singing in the rain, what a glorious feeling, I\'m happy again!',
                   'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
                   'I\'m gonna be the main event, like no king was before! I\'m brushing up on looking down, I\'m working on my ROAR!',
                   'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
                   'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
                   'I\'m singing in the rain! Just singing in the rain, what a glorious feeling, I\'m happy again!'],
         'col_2': ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
                   'I dreamed a dream in days gone by, when hope was high and life worth living',
                   'Red, the blood of angry men - black, the dark of ages past',
                   'do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
                   'Red, the blood of angry men - black, the dark of ages past',
                   'It was red and yellow and green and brown and scarlet and black and ochre and peach and ruby and olive and violet and fawn...']
         })
    y = [0, 1, 1, 0, 1, 0]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary')
    automl.search()
    assert automl.rankings['pipeline_name'][1:].str.contains('Text').all()


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_pipelines_per_batch(mock_fit, mock_score, X_y_binary):
    def total_pipelines(automl, num_batches, batch_size):
        total = 1 + len(automl.allowed_pipelines)
        total += ((num_batches - 1) * batch_size)
        return total

    X, y = X_y_binary

    # Checking for default of _pipelines_per_batch
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=2)
    automl.search()
    assert automl._pipelines_per_batch == 5
    assert automl._automl_algorithm.pipelines_per_batch == 5
    assert total_pipelines(automl, 2, 5) == len(automl.full_rankings)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=1, _pipelines_per_batch=2)
    automl.search()
    assert automl._pipelines_per_batch == 2
    assert automl._automl_algorithm.pipelines_per_batch == 2
    assert total_pipelines(automl, 1, 2) == len(automl.full_rankings)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=2, _pipelines_per_batch=10)
    automl.search()
    assert automl._pipelines_per_batch == 10
    assert automl._automl_algorithm.pipelines_per_batch == 10
    assert total_pipelines(automl, 2, 10) == len(automl.full_rankings)


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_respects_random_seed(mock_fit, mock_score, X_y_binary, dummy_classifier_estimator_class):

    X, y = X_y_binary

    class DummyPipeline(BinaryClassificationPipeline):
        component_graph = [dummy_classifier_estimator_class]
        num_pipelines_different_seed = 0
        num_pipelines_init = 0

        def __init__(self, parameters, random_seed=0):
            is_diff_random_seed = not (random_seed == 42)
            self.__class__.num_pipelines_init += 1
            self.__class__.num_pipelines_different_seed += is_diff_random_seed
            super().__init__(self.component_graph, parameters=parameters, random_seed=random_seed)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)
    pipelines = [DummyPipeline({})]
    DummyPipeline.num_pipelines_different_seed = 0
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", allowed_pipelines=pipelines,
                          random_seed=42, max_iterations=10)
    automl.search()
    assert DummyPipeline.num_pipelines_different_seed == 0 and DummyPipeline.num_pipelines_init


@pytest.mark.parametrize("callback", [log_error_callback, silent_error_callback, raise_error_callback])
@pytest.mark.parametrize("error_type", ['fit', "mean_cv_score", 'fit-single'])
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_error_callback(mock_fit, mock_score, error_type, callback, X_y_binary, caplog):
    X, y = X_y_binary
    if error_type == "mean_cv_score":
        msg = "Score Error!"
        mock_score.side_effect = Exception(msg)
    elif error_type == 'fit':
        mock_score.return_value = {"Log Loss Binary": 0.8}
        msg = 'all your model are belong to us'
        mock_fit.side_effect = Exception(msg)
    else:
        # throw exceptions for only one pipeline
        mock_score.return_value = {"Log Loss Binary": 0.8}
        msg = 'all your model are belong to us'
        mock_fit.side_effect = [Exception(msg)] * 3 + [None] * 100
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", error_callback=callback, train_best_pipeline=False, n_jobs=1)
    if callback in [log_error_callback, silent_error_callback]:
        exception = AutoMLSearchException
        match = "All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"
    else:
        exception = Exception
        match = msg

    if error_type == 'fit-single' and callback in [silent_error_callback, log_error_callback]:
        automl.search()
    else:
        with pytest.raises(exception, match=match):
            automl.search()

    if callback == silent_error_callback:
        assert msg not in caplog.text
    if callback == log_error_callback:
        assert f"Exception during automl search: {msg}" in caplog.text
        assert msg in caplog.text
    if callback in [raise_error_callback]:
        assert f"AutoML search raised a fatal exception: {msg}" in caplog.text
        assert msg in caplog.text


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_woodwork_user_types_preserved(mock_binary_fit, mock_binary_score,
                                              mock_multi_fit, mock_multi_score,
                                              mock_regression_fit, mock_regression_score, problem_type,
                                              X_y_binary, X_y_multi, X_y_regression):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        mock_fit = mock_binary_fit
        mock_score = mock_binary_score
        mock_score.return_value = {'Log Loss Binary': 1.0}

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        mock_fit = mock_multi_fit
        mock_score = mock_multi_score
        mock_score.return_value = {'Log Loss Multiclass': 1.0}

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        mock_fit = mock_regression_fit
        mock_score = mock_regression_score
        mock_score.return_value = {'R2': 1.0}

    X = pd.DataFrame(X)
    new_col = np.zeros(len(X))
    new_col[:int(len(new_col) / 2)] = 1
    X['cat col'] = pd.Series(new_col)
    X['num col'] = pd.Series(new_col)
    X['text col'] = pd.Series([f"{num}" for num in range(len(new_col))])
    X = ww.DataTable(X, semantic_tags={'cat col': 'category', 'num col': 'numeric'},
                     logical_types={'cat col': 'Categorical', 'num col': 'Integer', 'text col': 'NaturalLanguage'})
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type, max_batches=5)
    automl.search()
    for arg in mock_fit.call_args[0]:
        assert isinstance(arg, (ww.DataTable, ww.DataColumn))
        if isinstance(arg, ww.DataTable):
            assert arg.semantic_tags['cat col'] == {'category'}
            assert arg.logical_types['cat col'] == ww.logical_types.Categorical
            assert arg.semantic_tags['num col'] == {'numeric'}
            assert arg.logical_types['num col'] == ww.logical_types.Integer
            assert arg.semantic_tags['text col'] == set()
            assert arg.logical_types['text col'] == ww.logical_types.NaturalLanguage
    for arg in mock_score.call_args[0]:
        assert isinstance(arg, (ww.DataTable, ww.DataColumn))
        if isinstance(arg, ww.DataTable):
            assert arg.semantic_tags['cat col'] == {'category'}
            assert arg.logical_types['cat col'] == ww.logical_types.Categorical
            assert arg.semantic_tags['num col'] == {'numeric'}
            assert arg.logical_types['num col'] == ww.logical_types.Integer
            assert arg.semantic_tags['text col'] == set()
            assert arg.logical_types['text col'] == ww.logical_types.NaturalLanguage


def test_automl_validates_problem_configuration(X_y_binary):
    X, y = X_y_binary
    assert AutoMLSearch(X_train=X, y_train=y, problem_type="binary").problem_configuration == {}
    assert AutoMLSearch(X_train=X, y_train=y, problem_type="multiclass").problem_configuration == {}
    assert AutoMLSearch(X_train=X, y_train=y, problem_type="regression").problem_configuration == {}
    msg = "user_parameters must be a dict containing values for at least the date_index, gap, and max_delay parameters"
    with pytest.raises(ValueError, match=msg):
        AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression")
    with pytest.raises(ValueError, match=msg):
        AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression", problem_configuration={"gap": 3})
    with pytest.raises(ValueError, match=msg):
        AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression", problem_configuration={"max_delay": 2, "gap": 3})

    problem_config = AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression",
                                  problem_configuration={"date_index": "Date", "max_delay": 2, "gap": 3}).problem_configuration
    assert problem_config == {"date_index": "Date", "max_delay": 2, "gap": 3}


@patch('evalml.objectives.BinaryClassificationObjective.optimize_threshold')
def test_automl_best_pipeline(mock_optimize, X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', train_best_pipeline=False, n_jobs=1)
    automl.search()
    with pytest.raises(PipelineNotYetFittedError, match="not fitted"):
        automl.best_pipeline.predict(X)

    mock_optimize.return_value = 0.62

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', optimize_thresholds=False, objective="Accuracy Binary", n_jobs=1)
    automl.search()
    automl.best_pipeline.predict(X)
    assert automl.best_pipeline.threshold == 0.5

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', optimize_thresholds=True, objective="Log Loss Binary", n_jobs=1)
    automl.search()
    automl.best_pipeline.predict(X)
    assert automl.best_pipeline.threshold is None

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', optimize_thresholds=True, objective="Accuracy Binary", n_jobs=1)
    automl.search()
    automl.best_pipeline.predict(X)
    assert automl.best_pipeline.threshold == 0.62


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.RegressionPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
def test_automl_data_splitter_consistent(mock_binary_score, mock_binary_fit, mock_multi_score, mock_multi_fit,
                                         mock_regression_score, mock_regression_fit, problem_type,
                                         X_y_binary, X_y_multi, X_y_regression):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression

    data_splitters = []
    random_seed = [0, 0, 1]
    for seed in random_seed:
        a = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type, random_seed=seed, max_iterations=1)
        a.search()
        data_splitters.append([[set(train), set(test)] for train, test in a.data_splitter.split(X, y)])
    # append split from last random state again, should be referencing same datasplit object
    data_splitters.append([[set(train), set(test)] for train, test in a.data_splitter.split(X, y)])

    assert data_splitters[0] == data_splitters[1]
    assert data_splitters[1] != data_splitters[2]
    assert data_splitters[2] == data_splitters[3]


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_rerun(mock_fit, mock_score, X_y_binary, caplog):
    mock_score.return_value = {'Log Loss Binary': 1.0}
    msg = "AutoMLSearch.search() has already been run and will not run again on the same instance"
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", train_best_pipeline=False, n_jobs=1)
    automl.search()
    assert msg not in caplog.text
    automl.search()
    assert msg in caplog.text


@patch('evalml.pipelines.TimeSeriesRegressionPipeline.fit')
@patch('evalml.pipelines.TimeSeriesRegressionPipeline.score')
def test_timeseries_baseline_init_with_correct_gap_max_delay(mock_fit, mock_score, X_y_regression):

    X, y = X_y_regression
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression",
                          problem_configuration={"date_index": None, "gap": 6, "max_delay": 3}, max_iterations=1)
    automl.search()

    # Best pipeline is baseline pipeline because we only run one iteration
    assert automl.best_pipeline.parameters == {"pipeline": {"date_index": None, "gap": 6, "max_delay": 3},
                                               "Time Series Baseline Estimator": {"date_index": None, "gap": 6, "max_delay": 3}}


@pytest.mark.parametrize('problem_type', [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                                          ProblemTypes.TIME_SERIES_REGRESSION, ProblemTypes.REGRESSION])
def test_automl_does_not_include_positive_only_objectives_by_default(problem_type, X_y_regression):

    X, y = X_y_regression

    only_positive = []
    for name in get_all_objective_names():
        objective_class = get_objective(name)
        if objective_class.positive_only:
            only_positive.append(objective_class)

    search = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type,
                          problem_configuration={"date_index": None, 'gap': 0, 'max_delay': 0})
    assert search.objective not in only_positive
    assert all([obj not in only_positive for obj in search.additional_objectives])


@pytest.mark.parametrize('non_core_objective', get_non_core_objectives())
def test_automl_validate_objective(non_core_objective, X_y_regression):

    X, y = X_y_regression

    with pytest.raises(ValueError, match='is not allowed in AutoML!'):
        AutoMLSearch(X_train=X, y_train=y, problem_type=non_core_objective.problem_types[0],
                     objective=non_core_objective.name)

    with pytest.raises(ValueError, match='is not allowed in AutoML!'):
        AutoMLSearch(X_train=X, y_train=y, problem_type=non_core_objective.problem_types[0],
                     additional_objectives=[non_core_objective.name])


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_pipeline_params_simple(mock_fit, mock_score, X_y_binary):
    mock_score.return_value = {'Log Loss Binary': 1.0}
    X, y = X_y_binary
    params = {"Imputer": {"numeric_impute_strategy": "most_frequent"},
              "Logistic Regression Classifier": {"C": 20, "penalty": 'none'},
              "Elastic Net Classifier": {"alpha": 0.75, "l1_ratio": 0.2}}
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", pipeline_parameters=params, n_jobs=1)
    automl.search()
    for i, row in automl.rankings.iterrows():
        if 'Imputer' in row['parameters']:
            assert row['parameters']['Imputer']['numeric_impute_strategy'] == 'most_frequent'
        if 'Logistic Regression Classifier' in row['parameters']:
            assert row['parameters']['Logistic Regression Classifier']['C'] == 20
            assert row['parameters']['Logistic Regression Classifier']['penalty'] == 'none'
        if 'Elastic Net Classifier' in row['parameters']:
            assert row['parameters']['Elastic Net Classifier']['alpha'] == 0.75
            assert row['parameters']['Elastic Net Classifier']['l1_ratio'] == 0.2


@patch('evalml.pipelines.RegressionPipeline.fit')
@patch('evalml.pipelines.RegressionPipeline.score')
def test_automl_pipeline_params_multiple(mock_score, mock_fit, X_y_regression):
    mock_score.return_value = {'R2': 1.0}
    X, y = X_y_regression
    params = {'Imputer': {'numeric_impute_strategy': Categorical(['median', 'most_frequent'])},
              'Decision Tree Regressor': {'max_depth': Categorical([17, 18, 19]), 'max_features': Categorical(['auto'])},
              'Elastic Net Regressor': {"alpha": Real(0, 0.5), "l1_ratio": Categorical((0.01, 0.02, 0.03))}}
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', pipeline_parameters=params, n_jobs=1)
    automl.search()
    for i, row in automl.rankings.iterrows():
        if 'Imputer' in row['parameters']:
            assert row['parameters']['Imputer']['numeric_impute_strategy'] == Categorical(['median', 'most_frequent']).rvs(random_state=automl.random_seed)
        if 'Decision Tree Regressor' in row['parameters']:
            assert row['parameters']['Decision Tree Regressor']['max_depth'] == Categorical([17, 18, 19]).rvs(random_state=automl.random_seed)
            assert row['parameters']['Decision Tree Regressor']['max_features'] == 'auto'
        if 'Elastic Net Regressor' in row['parameters']:
            assert 0 < row['parameters']['Elastic Net Regressor']['alpha'] < 0.5
            assert row['parameters']['Elastic Net Regressor']['l1_ratio'] == Categorical((0.01, 0.02, 0.03)).rvs(random_state=automl.random_seed)


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.02})
def test_automl_respects_pipeline_parameters_with_duplicate_components(mock_score, mock_fit, X_y_binary):
    X, y = X_y_binary
    # Pass the input of the first imputer to the second imputer
    component_graph_dict = {"Imputer": ["Imputer"],
                            "Imputer_1": ["Imputer", "Imputer"],
                            "Random Forest Classifier": ["Random Forest Classifier", "Imputer_1"]}
    pipeline_dict = BinaryClassificationPipeline(component_graph_dict, custom_name="Pipeline from dict")

    component_graph_linear = ["Imputer", "Imputer", "Random Forest Classifier"]
    pipeline_linear = BinaryClassificationPipeline(component_graph_linear)
    automl = AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=[pipeline_dict, pipeline_linear],
                          pipeline_parameters={"Imputer": {"numeric_impute_strategy": Categorical(["most_frequent"])},
                                               "Imputer_1": {"numeric_impute_strategy": Categorical(["median"])}},
                          max_batches=3)
    automl.search()
    for i, row in automl.full_rankings.iterrows():
        if "Mode Baseline Binary" in row['pipeline_name']:
            continue
        assert row["parameters"]["Imputer"]["numeric_impute_strategy"] == "most_frequent"
        assert row["parameters"]["Imputer_1"]["numeric_impute_strategy"] == "median"

    component_graph_dict = {"One Hot Encoder": ["One Hot Encoder"],
                            "One Hot Encoder_1": ["One Hot Encoder", "One Hot Encoder"],
                            "Random Forest Classifier": ["Random Forest Classifier", "One Hot Encoder_1"]}
    pipeline_dict = BinaryClassificationPipeline(component_graph_dict, custom_name="Pipeline from dict")

    component_graph_linear = ["One Hot Encoder", "One Hot Encoder", "Random Forest Classifier"]
    pipeline_linear = BinaryClassificationPipeline(component_graph_linear)

    automl = AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=[pipeline_linear, pipeline_dict],
                          pipeline_parameters={"One Hot Encoder": {"top_n": Categorical([15])},
                                               "One Hot Encoder_1": {"top_n": Categorical([25])}},
                          max_batches=3)
    automl.search()
    for i, row in automl.full_rankings.iterrows():
        if "Mode Baseline Binary" in row['pipeline_name']:
            continue
        assert row["parameters"]["One Hot Encoder"]["top_n"] == 15
        assert row["parameters"]["One Hot Encoder_1"]["top_n"] == 25


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.02})
def test_automl_respects_pipeline_custom_hyperparameters_with_duplicate_components(mock_score, mock_fit, X_y_binary):
    X, y = X_y_binary

    custom_hyperparameters = {"Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", 'mean'])},
                              "Imputer_1": {"numeric_impute_strategy": Categorical(["median", 'mean'])},
                              "Random Forest Classifier": {"n_estimators": Categorical([50, 100])}}
    component_graph_dict = {"Imputer": ["Imputer"],
                            "Imputer_1": ["Imputer", "Imputer"],
                            "Random Forest Classifier": ["Random Forest Classifier", "Imputer_1"]}
    pipeline_dict = BinaryClassificationPipeline(component_graph_dict, custom_name="Pipeline from dict", custom_hyperparameters=custom_hyperparameters)

    custom_hyperparameters = {"Imputer": {"numeric_impute_strategy": Categorical(["mean"])},
                              "Imputer_1": {"numeric_impute_strategy": Categorical(["most_frequent", 'mean'])},
                              "Random Forest Classifier": {"n_estimators": Categorical([100, 125])}}
    component_graph_linear = ["Imputer", "Imputer", "Random Forest Classifier"]
    pipeline_linear = BinaryClassificationPipeline(component_graph_linear)

    automl = AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=[pipeline_dict, pipeline_linear],
                          max_batches=5)
    automl.search()
    for i, row in automl.full_rankings.iterrows():
        if "Mode Baseline Binary" in row['pipeline_name']:
            continue
        if row["pipeline_name"] == "Pipeline Dict":
            assert row["parameters"]["Imputer"]["numeric_impute_strategy"] in {"most_frequent", "mean"}
            assert row["parameters"]["Imputer_1"]["numeric_impute_strategy"] in {"median", "mean"}
            assert row["parameters"]["Random Forest Classifier"]["n_estimators"] in {50, 100}
        if row["pipeline_name"] == "Pipe Line Linear":
            assert row["parameters"]["Imputer"]["numeric_impute_strategy"] == "mean"
            assert row["parameters"]["Imputer_1"]["numeric_impute_strategy"] in {"most_frequent", "mean"}
            assert row["parameters"]["Random Forest Classifier"]["n_estimators"] in {100, 125}


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.02})
def test_automl_adds_pipeline_parameters_to_custom_pipeline_hyperparams(mock_score, mock_fit, X_y_binary):
    X, y = X_y_binary

    # Pass the input of the first imputer to the second imputer
    custom_hyperparameters = {"One Hot Encoder": {"top_n": Categorical([5, 10])}}

    component_graph = {"Imputer": ["Imputer"],
                       "Imputer_1": ["Imputer", "Imputer"],
                       "One Hot Encoder": ["One Hot Encoder", "Imputer_1"],
                       "Random Forest Classifier": ["Random Forest Classifier", "One Hot Encoder"]}
    pipeline_one = BinaryClassificationPipeline(component_graph, custom_name="Pipe Line One", custom_hyperparameters=custom_hyperparameters)
    pipeline_two = BinaryClassificationPipeline(["Imputer", "Imputer", "One Hot Encoder", "Random Forest Classifier"],
                                                custom_name="Pipe Line Two",
                                                custom_hyperparameters={"One Hot Encoder": {"top_n": Categorical([12, 10])}})

    pipeline_three = BinaryClassificationPipeline(["Imputer", "Imputer", "One Hot Encoder", "Random Forest Classifier"],
                                                  custom_name="Pipe Line Three",
                                                  custom_hyperparameters={"Imputer": {"numeric_imputer_strategy": Categorical(["median"])}})

    automl = AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=[pipeline_one, pipeline_two, pipeline_three],
                          pipeline_parameters={"Imputer": {"numeric_impute_strategy": Categorical(["most_frequent"])}},
                          max_batches=4)
    automl.search()

    expected_top_n = {"Pipe Line One": {5, 10}, "Pipe Line Two": {12, 10}, "Pipe Line Three": {10}}
    for i, row in automl.full_rankings.iterrows():
        if "Mode Baseline Binary" in row['pipeline_name']:
            continue
        assert row["parameters"]["Imputer"]["numeric_impute_strategy"] == "most_frequent"
        assert row["parameters"]["One Hot Encoder"]["top_n"] in expected_top_n[row["pipeline_name"]]
    assert any(row['parameters']["One Hot Encoder"]["top_n"] == 12 for _, row in automl.full_rankings.iterrows() if row["pipeline_name"] == "Pipe Line Two")
    assert any(row['parameters']["One Hot Encoder"]["top_n"] == 5 for _, row in automl.full_rankings.iterrows() if row["pipeline_name"] == "Pipe Line One")


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
def test_automl_pipeline_params_kwargs(mock_fit, mock_score, X_y_multi):
    mock_score.return_value = {'Log Loss Multiclass': 1.0}
    X, y = X_y_multi
    params = {'Imputer': {'numeric_impute_strategy': Categorical(['most_frequent'])},
              'Decision Tree Classifier': {'max_depth': Integer(1, 2), 'ccp_alpha': Real(0.1, 0.5)}}
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', pipeline_parameters=params,
                          allowed_model_families=[ModelFamily.DECISION_TREE], n_jobs=1)
    automl.search()
    for i, row in automl.rankings.iterrows():
        if 'Imputer' in row['parameters']:
            assert row['parameters']['Imputer']['numeric_impute_strategy'] == 'most_frequent'
        if 'Decision Tree Classifier' in row['parameters']:
            assert 0.1 < row['parameters']['Decision Tree Classifier']['ccp_alpha'] < 0.5
            assert row['parameters']['Decision Tree Classifier']['max_depth'] == 1


@pytest.mark.parametrize("random_seed", [0, 1, 9])
@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
def test_automl_pipeline_random_seed(mock_fit, mock_score, random_seed, X_y_multi):
    mock_score.return_value = {'Log Loss Multiclass': 1.0}
    X, y = X_y_multi
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', random_seed=random_seed, n_jobs=1)
    automl.search()

    for i, row in automl.rankings.iterrows():
        if 'Base' not in list(row['parameters'].keys())[0]:
            assert automl.get_pipeline(row['id']).random_seed == random_seed


@pytest.mark.parametrize("ensembling", [True, False])
@pytest.mark.parametrize("ensemble_split_size", [0.1, 0.2])
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.3})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_ensembling_training(mock_fit, mock_score, ensemble_split_size, ensembling, X_y_binary):
    X, y = X_y_binary
    # don't train the best pipeline since we check usage of the ensembling CV through the .fit mock
    ensemble_pipelines = len(get_estimators("binary")) + 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', random_seed=0, n_jobs=1, max_batches=ensemble_pipelines, ensembling=ensembling,
                          train_best_pipeline=False, _ensembling_split_size=ensemble_split_size)
    automl.search()
    training_indices, ensembling_indices, _, _ = split_data(ww.DataTable(np.arange(X.shape[0])), y, problem_type='binary', test_size=ensemble_split_size, random_seed=0)
    training_indices, ensembling_indices = training_indices.to_dataframe()[0].tolist(), ensembling_indices.to_dataframe()[0].tolist()
    if ensembling:
        assert automl.ensembling
        # check that the X_train data is all used for the length
        assert len(training_indices) == (len(mock_fit.call_args_list[-2][0][0]) + len(mock_score.call_args_list[-2][0][0]))
        # last call will be the stacking ensembler
        assert len(ensembling_indices) == (len(mock_fit.call_args_list[-1][0][0]) + len(mock_score.call_args_list[-1][0][0]))
    else:
        # verify that there is no creation of ensembling CV data
        assert not automl.ensembling_indices
        for i in [-1, -2]:
            assert len(X) == (len(mock_fit.call_args_list[i][0][0]) + len(mock_score.call_args_list[i][0][0]))


@pytest.mark.parametrize("best_pipeline", [-1, -2])
@pytest.mark.parametrize("ensemble_split_size", [0.1, 0.2, 0.5])
@pytest.mark.parametrize("indices", [[i for i in range(100)], [f"index_{i}" for i in range(100)]])
@patch('evalml.automl.automl_search.AutoMLSearch.rankings', new_callable=PropertyMock)
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.3})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_ensembling_best_pipeline(mock_fit, mock_score, mock_rankings, indices, ensemble_split_size, best_pipeline, X_y_binary, has_minimal_dependencies):
    X, y = X_y_binary
    X = pd.DataFrame(X, index=indices)
    y = pd.Series(y, index=indices)
    ensemble_pipelines = len(get_estimators("binary")) + 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', random_seed=0, n_jobs=1, max_batches=ensemble_pipelines,
                          ensembling=True, _ensembling_split_size=ensemble_split_size)
    ensembling_num = (1 + len(automl.allowed_pipelines) + len(automl.allowed_pipelines) * automl._pipelines_per_batch + 1) + best_pipeline
    mock_rankings.return_value = pd.DataFrame({"id": ensembling_num, "pipeline_name": "stacked_ensembler", "mean_cv_score": 0.1}, index=[0])
    automl.search()
    # when best_pipeline == -1, model is ensembling,
    # otherwise, the model is a different model
    # the ensembling_num formula is taken from AutoMLSearch
    if best_pipeline == -1:
        assert automl.best_pipeline.model_family == ModelFamily.ENSEMBLE
    else:
        assert automl.best_pipeline.model_family != ModelFamily.ENSEMBLE
    assert len(mock_fit.call_args_list[-1][0][0]) == len(X)
    assert len(mock_fit.call_args_list[-1][0][1]) == len(y)


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.3})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_no_ensembling_best_pipeline(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    # does not ensemble
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', random_seed=0, n_jobs=1, max_iterations=2)
    automl.search()
    assert len(mock_fit.call_args_list[-1][0][0]) == len(X)
    assert len(mock_fit.call_args_list[-1][0][1]) == len(y)


@pytest.mark.parametrize("ensemble_split_size", [-1, 0, 1.0, 1.1])
def test_automl_ensemble_split_size(ensemble_split_size, X_y_binary):
    X, y = X_y_binary
    ensemble_pipelines = len(get_estimators("binary")) + 2
    with pytest.raises(ValueError, match="Ensembling split size must be between"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', random_seed=0, ensembling=True, max_batches=ensemble_pipelines, _ensembling_split_size=ensemble_split_size)


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.3})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_best_pipeline_feature_types_ensembling(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X['text column'] = ["Here is a text column that we want to treat as categorical if possible, but we want it to have some unique {} value".format(i % 10) for i in range(len(X))]
    X = ww.DataTable(X, logical_types={1: "categorical", "text column": "categorical"})
    y = ww.DataColumn(pd.Series(y))
    ensemble_pipelines = len(get_estimators("binary")) + 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', random_seed=0, n_jobs=1, max_batches=ensemble_pipelines, ensembling=True,
                          train_best_pipeline=True)
    assert automl.ensembling
    automl.search()
    # ensure we use the full X data for training the best pipeline, which isn't ensembling pipeline
    assert len(X) == len(mock_fit.call_args_list[-1][0][0])
    # check that the logical types were preserved
    assert str(mock_fit.call_args_list[-1][0][0].logical_types[1]) == 'Categorical'
    assert str(mock_fit.call_args_list[-1][0][0].logical_types['text column']) == 'Categorical'


def test_automl_check_for_high_variance(X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary')
    cv_scores = pd.Series([1, 1, 1])
    pipeline = dummy_binary_pipeline_class(parameters={})
    assert not automl._check_for_high_variance(pipeline, cv_scores.mean(), cv_scores.std())

    cv_scores = pd.Series([0, 0, 0])
    assert not automl._check_for_high_variance(pipeline, cv_scores.mean(), cv_scores.std())

    cv_scores = pd.Series([0, 1, np.nan, np.nan])
    assert automl._check_for_high_variance(pipeline, cv_scores.mean(), cv_scores.std())

    cv_scores = pd.Series([0, 1, 2, 3])
    assert automl._check_for_high_variance(pipeline, cv_scores.mean(), cv_scores.std())

    cv_scores = pd.Series([0, -1, -1, -1])
    assert automl._check_for_high_variance(pipeline, cv_scores.mean(), cv_scores.std())


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_check_high_variance_logs_warning(mock_fit_binary, X_y_binary, caplog):
    X, y = X_y_binary

    with patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 1}):
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary')
        automl.search()
        out = caplog.text
        assert "High coefficient of variation" not in out

    caplog.clear()

    desired_score_values = [{"Log Loss Binary": i} for i in [1, 2, 10] * 2]
    with patch('evalml.pipelines.BinaryClassificationPipeline.score', side_effect=desired_score_values):
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=2)
        automl.search()
        out = caplog.text
        assert "High coefficient of variation" in out


def test_automl_raises_error_with_duplicate_pipeline_names(X_y_binary):
    X, y = X_y_binary
    pipeline_1 = BinaryClassificationPipeline(component_graph=["Imputer", "Random Forest Classifier"], custom_name="Custom Pipeline")
    pipeline_2 = BinaryClassificationPipeline(component_graph=["Imputer", "Logistic Regression Classifier"], custom_name="Custom Pipeline")
    pipeline_3 = BinaryClassificationPipeline(component_graph=["Logistic Regression Classifier"], custom_name="My Pipeline 3")
    pipeline_4 = BinaryClassificationPipeline(component_graph=["Random Forest Classifier"], custom_name="My Pipeline 3")

    with pytest.raises(ValueError,
                       match="All pipeline names must be unique. The name 'Custom Pipeline' was repeated."):
        AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=[pipeline_1, pipeline_2, pipeline_3])

    with pytest.raises(ValueError,
                       match="All pipeline names must be unique. The names 'Custom Pipeline', 'My Pipeline 3' were repeated."):
        AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=[pipeline_1, pipeline_2, pipeline_3, pipeline_4])


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_train_batch_score_batch(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):

    def make_dummy_pipeline(index):
        class Pipeline(dummy_binary_pipeline_class):
            custom_name = f"Pipeline {index}"
        return Pipeline({})

    pipelines = [make_dummy_pipeline(i) for i in range(3)]

    X, y = X_y_binary

    mock_score.return_value = {"Log Loss Binary": 0.1}
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=3)
    automl.search()

    mock_fit.side_effect = [None, Exception("foo"), None]
    fitted_pipelines = automl.train_pipelines(pipelines)
    assert fitted_pipelines.keys() == {"Pipeline 0", "Pipeline 2"}

    score_effects = [{"Log Loss Binary": 0.1}, {"Log Loss Binary": 0.2}, {"Log Loss Binary": 0.3}]
    mock_score.side_effect = score_effects
    expected_scores = {f"Pipeline {i}": effect for i, effect in zip(range(3), score_effects)}
    scores = automl.score_pipelines(pipelines, X, y, ["Log Loss Binary"])
    assert scores == expected_scores


def test_train_batch_returns_trained_pipelines(X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")
    rf_pipeline = BinaryClassificationPipeline(["Random Forest Classifier"], parameters={"Random Forest Classifier": {"n_jobs": 1}})
    lrc_pipeline = BinaryClassificationPipeline(["Logistic Regression Classifier"], parameters={"Logistic Regression Classifier": {"n_jobs": 1}})

    pipelines = [rf_pipeline, lrc_pipeline]
    fitted_pipelines = automl.train_pipelines(pipelines)

    assert all([isinstance(pl, PipelineBase) for pl in fitted_pipelines.values()])

    # Check that the output pipelines are fitted but the input pipelines are not
    for original_pipeline in pipelines:
        fitted_pipeline = fitted_pipelines[original_pipeline.name]
        assert fitted_pipeline.name == original_pipeline.name
        assert fitted_pipeline._is_fitted
        assert fitted_pipeline != original_pipeline
        assert fitted_pipeline.parameters == original_pipeline.parameters


@pytest.mark.parametrize("pipeline_fit_side_effect",
                         [[None] * 6, [None, Exception("foo"), None],
                          [None, Exception("bar"), Exception("baz")],
                          [Exception("Everything"), Exception("is"), Exception("broken")]])
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.3})
def test_train_batch_works(mock_score, pipeline_fit_side_effect, X_y_binary,
                           dummy_binary_pipeline_class, stackable_classifiers, caplog):

    exceptions_to_check = [str(e) for e in pipeline_fit_side_effect if isinstance(e, Exception)]

    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_iterations=2,
                          train_best_pipeline=False, n_jobs=1)

    def make_pipeline_name(index):
        class DummyPipeline(dummy_binary_pipeline_class):
            custom_name = f"Pipeline {index}"
        return DummyPipeline({'Mock Classifier': {'a': index}})

    pipelines = [make_pipeline_name(i) for i in range(len(pipeline_fit_side_effect) - 1)]
    input_pipelines = [BinaryClassificationPipeline([classifier]) for classifier in stackable_classifiers[:2]]
    ensemble = BinaryClassificationPipeline([StackedEnsembleClassifier],
                                            parameters={"Stacked Ensemble Classifier": {"input_pipelines": input_pipelines, "n_jobs": 1}})
    pipelines.append(ensemble)

    def train_batch_and_check():
        caplog.clear()
        with patch('evalml.pipelines.BinaryClassificationPipeline.fit') as mock_fit:
            mock_fit.side_effect = pipeline_fit_side_effect

            trained_pipelines = automl.train_pipelines(pipelines)

            assert len(trained_pipelines) == len(pipeline_fit_side_effect) - len(exceptions_to_check)
            assert mock_fit.call_count == len(pipeline_fit_side_effect)
            for exception in exceptions_to_check:
                assert exception in caplog.text

    # Test training before search is run
    train_batch_and_check()

    # Test training after search.
    automl.search()

    train_batch_and_check()


no_exception_scores = {"F1": 0.9, "AUC": 0.7, "Log Loss Binary": 0.25}


@pytest.mark.parametrize("pipeline_score_side_effect",
                         [[no_exception_scores] * 6,
                          [no_exception_scores,
                           PipelineScoreError(exceptions={"AUC": (Exception(), []), "Log Loss Binary": (Exception(), [])},
                                              scored_successfully={"F1": 0.2}),
                           no_exception_scores],
                          [no_exception_scores,
                           PipelineScoreError(exceptions={"AUC": (Exception(), []), "Log Loss Binary": (Exception(), [])},
                                              scored_successfully={"F1": 0.3}),
                           PipelineScoreError(exceptions={"AUC": (Exception(), []), "F1": (Exception(), [])},
                                              scored_successfully={"Log Loss Binary": 0.2})],
                          [PipelineScoreError(exceptions={"Log Loss Binary": (Exception(), []), "F1": (Exception(), [])},
                                              scored_successfully={"AUC": 0.6}),
                           PipelineScoreError(exceptions={"AUC": (Exception(), []), "Log Loss Binary": (Exception(), [])},
                                              scored_successfully={"F1": 0.2}),
                           PipelineScoreError(exceptions={"Log Loss Binary": (Exception(), [])},
                                              scored_successfully={"AUC": 0.2, "F1": 0.1})]])
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
def test_score_batch_works(mock_score, pipeline_score_side_effect, X_y_binary,
                           dummy_binary_pipeline_class, stackable_classifiers, caplog):

    exceptions_to_check = []
    expected_scores = {}
    for i, e in enumerate(pipeline_score_side_effect):
        # Ensemble pipeline has different name
        pipeline_name = f"Pipeline {i}" if i < len(pipeline_score_side_effect) - 1 else "Templated Pipeline"
        scores = no_exception_scores
        if isinstance(e, PipelineScoreError):
            scores = {"F1": np.nan, "AUC": np.nan, "Log Loss Binary": np.nan}
            scores.update(e.scored_successfully)
            exceptions_to_check.append(f"Score error for {pipeline_name}")

        expected_scores[pipeline_name] = scores

    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1,
                          allowed_pipelines=[dummy_binary_pipeline_class({})])

    def make_pipeline_name(index):
        class DummyPipeline(dummy_binary_pipeline_class):
            custom_name = f"Pipeline {index}"
        return DummyPipeline({'Mock Classifier': {'a': index}})

    pipelines = [make_pipeline_name(i) for i in range(len(pipeline_score_side_effect) - 1)]
    input_pipelines = [BinaryClassificationPipeline([classifier]) for classifier in stackable_classifiers[:2]]
    ensemble = BinaryClassificationPipeline([StackedEnsembleClassifier],
                                            parameters={"Stacked Ensemble Classifier": {"input_pipelines": input_pipelines, "n_jobs": 1}},
                                            custom_name="Templated Pipeline")
    pipelines.append(ensemble)

    def score_batch_and_check():
        caplog.clear()
        with patch('evalml.pipelines.BinaryClassificationPipeline.score') as mock_score:
            mock_score.side_effect = pipeline_score_side_effect

            scores = automl.score_pipelines(pipelines, X, y, objectives=["Log Loss Binary", "F1", "AUC"])
            assert scores == expected_scores
            for exception in exceptions_to_check:
                assert exception in caplog.text

    # Test scoring before search
    score_batch_and_check()

    automl.search()

    # Test scoring after search
    score_batch_and_check()


def test_train_pipelines_score_pipelines_raise_exception_with_duplicate_names(X_y_binary, dummy_binary_pipeline_class):

    class Pipeline1(dummy_binary_pipeline_class):
        custom_name = "My Pipeline"

    class Pipeline2(dummy_binary_pipeline_class):
        custom_name = "My Pipeline"

    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1,
                          allowed_pipelines=[dummy_binary_pipeline_class({})])

    with pytest.raises(ValueError, match="All pipeline names must be unique. The name 'My Pipeline' was repeated."):
        automl.train_pipelines([Pipeline2({}), Pipeline1({})])

    with pytest.raises(ValueError, match="All pipeline names must be unique. The name 'My Pipeline' was repeated."):
        automl.score_pipelines([Pipeline2({}), Pipeline1({})], None, None, None)


def test_score_batch_before_fitting_yields_error_nan_scores(X_y_binary, dummy_binary_pipeline_class, caplog):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1,
                          allowed_pipelines=[dummy_binary_pipeline_class({})])

    scored_pipelines = automl.score_pipelines([dummy_binary_pipeline_class({})], X, y,
                                              objectives=["Log Loss Binary", F1()])
    assert scored_pipelines == {"Mock Binary Classification Pipeline": {"Log Loss Binary": np.nan,
                                                                        "F1": np.nan}}

    assert "Score error for Mock Binary Classification Pipeline" in caplog.text
    assert "This LabelEncoder instance is not fitted yet." in caplog.text


def test_high_cv_check_no_warning_for_divide_by_zero(X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")
    with pytest.warns(None) as warnings:
        automl._check_for_high_variance(dummy_binary_pipeline_class({}), cv_mean=np.array([0.0]),
                                        cv_std=np.array([0.1]))
    assert len(warnings) == 0

    with pytest.warns(None) as warnings:
        # mean is 0 but std is not
        automl._check_for_high_variance(dummy_binary_pipeline_class({}),
                                        cv_mean=np.array([0.0, 1.0, -1.0]).mean(), cv_std=np.array([0.0, 1.0, -1.0]).std())
    assert len(warnings) == 0


@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
@patch('evalml.pipelines.RegressionPipeline.score', return_value={"R2": 0.3})
@patch('evalml.pipelines.ClassificationPipeline.score', return_value={"Log Loss Multiclass": 0.3})
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.3})
@patch('evalml.automl.engine.sequential_engine.train_pipeline')
def test_automl_supports_float_targets_for_classification(mock_train, mock_binary_score, mock_multi_score, mock_regression_score,
                                                          automl_type, X_y_binary, X_y_multi, X_y_regression,
                                                          dummy_binary_pipeline_class,
                                                          dummy_regression_pipeline_class,
                                                          dummy_multiclass_pipeline_class):
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        y = pd.Series(y).map({0: -5.19, 1: 6.7})
        mock_train.return_value = dummy_binary_pipeline_class({})
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        y = pd.Series(y).map({0: -5.19, 1: 6.7, 2: 2.03})
        mock_train.return_value = dummy_multiclass_pipeline_class({})
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        y = pd.Series(y)
        mock_train.return_value = dummy_regression_pipeline_class({})

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, random_seed=0, n_jobs=1)
    automl.search()

    # Assert that we train pipeline on the original target, not the encoded one used in EngineBase for data splitting
    _, kwargs = mock_train.call_args
    mock_y = kwargs["y"]
    pd.testing.assert_series_equal(mock_y.to_series(), y, check_dtype=False)


@pytest.mark.parametrize("problem_type", [ProblemTypes.TIME_SERIES_REGRESSION, ProblemTypes.TIME_SERIES_BINARY,
                                          ProblemTypes.TIME_SERIES_MULTICLASS])
def test_automl_issues_beta_warning_for_time_series(problem_type, X_y_binary):

    X, y = X_y_binary

    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter("always")
        AutoMLSearch(X, y, problem_type=problem_type, problem_configuration={"date_index": None, "gap": 0, "max_delay": 2})
        assert len(warn) == 1
        message = "Time series support in evalml is still in beta, which means we are still actively building its core features"
        assert str(warn[0].message).startswith(message)


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.3})
@patch('evalml.automl.engine.sequential_engine.train_pipeline')
def test_automl_drop_index_columns(mock_train, mock_binary_score, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X['index_col'] = pd.Series(range(len(X)))
    X = ww.DataTable(X)
    X = X.set_index('index_col')

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=2)
    automl.search()
    for pipeline in automl.allowed_pipelines:
        assert pipeline.get_component('Drop Columns Transformer')
        assert 'Drop Columns Transformer' in pipeline.hyperparameters
        assert pipeline.hyperparameters['Drop Columns Transformer'] == {}

    all_drop_column_params = []
    for _, row in automl.full_rankings.iterrows():
        if "Baseline" not in row.pipeline_name:
            all_drop_column_params.append(row.parameters['Drop Columns Transformer']['columns'])
    assert all(param == ['index_col'] for param in all_drop_column_params)


def test_automl_validates_data_passed_in_to_allowed_pipelines(X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary

    with pytest.raises(ValueError, match="Parameter allowed_pipelines must be either None or a list!"):
        AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=dummy_binary_pipeline_class)

    with pytest.raises(ValueError, match="Every element of allowed_pipelines must an instance of PipelineBase!"):
        AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=[dummy_binary_pipeline_class])

    with pytest.raises(ValueError, match="Every element of allowed_pipelines must an instance of PipelineBase!"):
        AutoMLSearch(X, y, problem_type="binary", allowed_pipelines=[dummy_binary_pipeline_class.custom_name, dummy_binary_pipeline_class])


@pytest.mark.parametrize("problem_type", [problem_type for problem_type in ProblemTypes.all_problem_types if not is_time_series(problem_type)])
def test_automl_baseline_pipeline_predictions_and_scores(problem_type):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    y = pd.Series([10, 11, 10, 10])
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([10, 11, 12, 11])
    automl = AutoMLSearch(X, y, problem_type=problem_type)
    baseline = automl._get_baseline_pipeline()
    baseline.fit(X, y)

    if problem_type == ProblemTypes.BINARY:
        expected_predictions = pd.Series(np.array([10] * len(X)), dtype="Int64")
        expected_predictions_proba = pd.DataFrame({10: [1., 1., 1., 1.], 11: [0., 0., 0., 0.]})
    if problem_type == ProblemTypes.MULTICLASS:
        expected_predictions = pd.Series(np.array([11] * len(X)), dtype="Int64")
        expected_predictions_proba = pd.DataFrame({10: [0., 0., 0., 0.], 11: [1., 1., 1., 1.], 12: [0., 0., 0., 0.]})
    if problem_type == ProblemTypes.REGRESSION:
        mean = y.mean()
        expected_predictions = pd.Series([mean] * len(X))

    pd.testing.assert_series_equal(expected_predictions, baseline.predict(X).to_series())
    if is_classification(problem_type):
        pd.testing.assert_frame_equal(expected_predictions_proba, baseline.predict_proba(X).to_dataframe())
    np.testing.assert_allclose(baseline.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))


@pytest.mark.parametrize('gap', [0, 1])
@pytest.mark.parametrize("problem_type", [problem_type for problem_type in ProblemTypes.all_problem_types if is_time_series(problem_type)])
def test_automl_baseline_pipeline_predictions_and_scores_time_series(problem_type, gap):
    X = pd.DataFrame({"a": [4, 5, 6, 7, 8]})
    y = pd.Series([0, 1, 1, 0, 1])
    expected_predictions_proba = pd.DataFrame({0: pd.Series([1, 0, 0, 1, 0], dtype="float64"),
                                               1: pd.Series([0, 1, 1, 0, 1], dtype="float64")})
    if problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        y = pd.Series([0, 1, 2, 2, 1])
        expected_predictions_proba = pd.DataFrame({0: pd.Series([1, 0, 0, 0, 0], dtype="float64"),
                                                   1: pd.Series([0, 1, 0, 0, 1], dtype="float64"),
                                                   2: pd.Series([0, 0, 1, 1, 0], dtype="float64")})
    if gap == 0:
        # Shift to pad the first row with Nans
        expected_predictions_proba = expected_predictions_proba.shift(1)

    automl = AutoMLSearch(X, y,
                          problem_type=problem_type,
                          problem_configuration={"date_index": None, "gap": gap, "max_delay": 1})
    baseline = automl._get_baseline_pipeline()
    baseline.fit(X, y)

    expected_predictions = y.shift(1) if gap == 0 else y
    expected_predictions = expected_predictions.reset_index(drop=True)
    if not expected_predictions.isnull().values.any():
        expected_predictions = expected_predictions.astype("Int64")

    pd.testing.assert_series_equal(expected_predictions, baseline.predict(X, y).to_series())
    if is_classification(problem_type):
        pd.testing.assert_frame_equal(expected_predictions_proba, baseline.predict_proba(X, y).to_dataframe())
    np.testing.assert_allclose(baseline.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))
