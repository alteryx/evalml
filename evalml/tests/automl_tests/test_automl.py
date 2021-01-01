import os
from itertools import product
from unittest.mock import MagicMock, patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn.model_selection import KFold, StratifiedKFold

from evalml import AutoMLSearch
from evalml.automl.callbacks import (
    log_and_save_error_callback,
    log_error_callback,
    raise_and_save_error_callback,
    raise_error_callback,
    silent_error_callback
)
from evalml.automl.utils import (
    _LARGE_DATA_PERCENT_VALIDATION,
    _LARGE_DATA_ROW_THRESHOLD
)
from evalml.data_checks import (
    DataCheck,
    DataCheckError,
    DataChecks,
    DataCheckWarning
)
from evalml.demos import load_breast_cancer, load_wine
from evalml.exceptions import (
    AutoMLSearchException,
    PipelineNotFoundError,
    PipelineNotYetFittedError
)
from evalml.model_family import ModelFamily
from evalml.objectives import CostBenefitMatrix, FraudCost
from evalml.objectives.utils import get_core_objectives, get_objective
from evalml.pipelines import (
    BinaryClassificationPipeline,
    Estimator,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesRegressionPipeline
)
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.preprocessing.data_splitters import (
    TimeSeriesSplit,
    TrainingValidationSplit
)
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.tuners import NoParamsException, RandomSearchTuner
from evalml.utils.gen_utils import (
    categorical_dtypes,
    check_random_state_equality,
    get_random_state,
    numeric_and_boolean_dtypes
)


@pytest.mark.parametrize("automl_type", [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_search_results(X_y_regression, X_y_binary, X_y_multi, automl_type):
    expected_cv_data_keys = {'all_objective_scores', 'score', 'binary_classification_threshold'}
    if automl_type == ProblemTypes.REGRESSION:
        expected_pipeline_class = RegressionPipeline
        X, y = X_y_regression
    elif automl_type == ProblemTypes.BINARY:
        expected_pipeline_class = BinaryClassificationPipeline
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        expected_pipeline_class = MulticlassClassificationPipeline
        X, y = X_y_multi

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, max_iterations=2, n_jobs=1)
    automl.search()
    assert automl.results.keys() == {'pipeline_results', 'search_order', 'errors'}
    assert automl.results['search_order'] == [0, 1]
    assert len(automl.results['pipeline_results']) == 2
    for pipeline_id, results in automl.results['pipeline_results'].items():
        assert results.keys() == {'id', 'pipeline_name', 'pipeline_class', 'pipeline_summary', 'parameters', 'score', 'high_variance_cv', 'training_time',
                                  'cv_data', 'percent_better_than_baseline_all_objectives',
                                  'percent_better_than_baseline', 'validation_score'}
        assert results['id'] == pipeline_id
        assert isinstance(results['pipeline_name'], str)
        assert issubclass(results['pipeline_class'], expected_pipeline_class)
        assert isinstance(results['pipeline_summary'], str)
        assert isinstance(results['parameters'], dict)
        assert isinstance(results['score'], float)
        assert isinstance(results['high_variance_cv'], bool)
        assert isinstance(results['cv_data'], list)
        for cv_result in results['cv_data']:
            assert cv_result.keys() == expected_cv_data_keys
            if automl_type == ProblemTypes.BINARY:
                assert isinstance(cv_result['binary_classification_threshold'], float)
            else:
                assert cv_result['binary_classification_threshold'] is None
            all_objective_scores = cv_result["all_objective_scores"]
            for score in all_objective_scores.values():
                assert score is not None
        assert automl.get_pipeline(pipeline_id).parameters == results['parameters']
        assert results['validation_score'] == pd.Series([fold['score'] for fold in results['cv_data']])[0]
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)
    assert np.all(automl.rankings.dtypes == pd.Series(
        [np.dtype('int64'), np.dtype('O'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('O')],
        index=['id', 'pipeline_name', 'score', "validation_score", 'percent_better_than_baseline', 'high_variance_cv', 'parameters']))
    assert np.all(automl.full_rankings.dtypes == pd.Series(
        [np.dtype('int64'), np.dtype('O'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64'), np.dtype('bool'), np.dtype('O')],
        index=['id', 'pipeline_name', 'score', "validation_score", 'percent_better_than_baseline', 'high_variance_cv', 'parameters']))


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
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_str_search(mock_fit, mock_score, mock_predict_proba, mock_optimize_threshold, X_y_binary):
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
        'data_splitter': StratifiedKFold(5),
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
        'Random State': 'RandomState(MT19937)',
        'n_jobs': search_params['n_jobs'],
        'Optimize Thresholds': search_params['optimize_thresholds']
    }

    automl = AutoMLSearch(X_train=X, y_train=y, **search_params)
    mock_score.return_value = {automl.objective.name: 1.0}
    mock_optimize_threshold.return_value = 0.62
    str_rep = str(automl)
    for param, value in param_str_reps.items():
        if isinstance(value, list):
            assert f"{param}" in str_rep
            for item in value:
                assert f"\t{str(item)}" in str_rep
        else:
            assert f"{param}: {str(value)}" in str_rep
    assert "Search Results" not in str_rep

    mock_score.return_value = {automl.objective.name: 1.0}
    automl.search()
    mock_fit.assert_called()
    mock_score.assert_called()
    mock_predict_proba.assert_called()
    mock_optimize_threshold.assert_called()

    str_rep = str(automl)
    assert "Search Results:" in str_rep
    assert automl.rankings.drop(['parameters'], axis='columns').to_string() in str_rep


def test_automl_data_check_results_is_none_before_search(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1, n_jobs=1)
    assert automl.data_check_results is None


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_empty_data_checks(mock_fit, mock_score):
    X = pd.DataFrame({"feature1": [1, 2, 3],
                      "feature2": [None, None, None]})
    y = pd.Series([1, 1, 1])

    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=1)
    automl.search(data_checks=[])
    assert automl.data_check_results == {"warnings": [], "errors": []}
    mock_fit.assert_called()
    mock_score.assert_called()

    automl.search(data_checks="disabled")
    assert automl.data_check_results == {"warnings": [], "errors": []}

    automl.search(data_checks=None)
    assert automl.data_check_results == {"warnings": [], "errors": []}


@patch('evalml.data_checks.DefaultDataChecks.validate')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_default_data_checks(mock_fit, mock_score, mock_validate, X_y_binary, caplog):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}
    mock_validate.return_value = {
        "warnings": [DataCheckWarning("default data check warning", "DefaultDataChecks")],
        "errors": []
    }

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)
    automl.search()
    out = caplog.text
    assert "default data check warning" in out
    assert automl.data_check_results == mock_validate.return_value
    mock_fit.assert_called()
    mock_score.assert_called()
    mock_validate.assert_called()


class MockDataCheckErrorAndWarning(DataCheck):
    def validate(self, X, y):
        return {
            "warnings": [],
            "errors": [DataCheckError("error one", self.name), DataCheckWarning("warning one", self.name)]
        }


@pytest.mark.parametrize("data_checks",
                         [[MockDataCheckErrorAndWarning()],
                          DataChecks([MockDataCheckErrorAndWarning])])
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_data_checks_raises_error(mock_fit, mock_score, data_checks, caplog):
    X = pd.DataFrame()
    y = pd.Series()

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=1)

    with pytest.raises(ValueError, match="Data checks raised"):
        automl.search(data_checks=data_checks)

    out = caplog.text
    assert "error one" in out
    assert "warning one" in out
    assert automl.data_check_results == MockDataCheckErrorAndWarning().validate(X, y)


def test_automl_bad_data_check_parameter_type():
    X = pd.DataFrame()
    y = pd.Series()

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=1, n_jobs=1)

    with pytest.raises(ValueError, match="Parameter data_checks must be a list. Received int."):
        automl.search(data_checks=1)
    with pytest.raises(ValueError, match="All elements of parameter data_checks must be an instance of DataCheck."):
        automl.search(data_checks=[1])
    with pytest.raises(ValueError, match="If data_checks is a string, it must be either 'auto' or 'disabled'. "
                                         "Received 'default'."):
        automl.search(data_checks="default")
    with pytest.raises(ValueError, match="All elements of parameter data_checks must be an instance of DataCheck."):
        automl.search(data_checks=[DataChecks([]), 1])
    with pytest.raises(ValueError, match="All elements of parameter data_checks must be an instance of DataCheck."):
        automl.search(data_checks=[MockDataCheckErrorAndWarning])


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
        'Data Splitting': 'StratifiedKFold(n_splits=3, random_state=0, shuffle=True)',
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
        'Random State': 'RandomState(MT19937)',
        'n_jobs': '-1',
        'Verbose': 'True',
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

        def fit(self, X, y):
            """Mock fit, noop"""

    allowed_pipelines = [MockFeatureSelectionPipeline]
    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=2, start_iteration_callback=start_iteration_callback, allowed_pipelines=allowed_pipelines)
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
    assert automl.data_check_results == {"warnings": [], "errors": []}
    mock_fit.assert_called()
    mock_score.assert_called()
    assert mock_algo_next_batch.call_count == 1
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    assert pipeline_results[0].get('score') == 1.0


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.__init__')
def test_automl_allowed_pipelines_algorithm(mock_algo_init, dummy_binary_pipeline_class, X_y_binary):
    mock_algo_init.side_effect = Exception('mock algo init')
    X, y = X_y_binary

    allowed_pipelines = [dummy_binary_pipeline_class]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', allowed_pipelines=allowed_pipelines, max_iterations=10)
    with pytest.raises(Exception, match='mock algo init'):
        automl.search()
    assert mock_algo_init.call_count == 1
    _, kwargs = mock_algo_init.call_args
    assert kwargs['max_iterations'] == 10
    assert kwargs['allowed_pipelines'] == allowed_pipelines

    allowed_model_families = [ModelFamily.RANDOM_FOREST]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', allowed_model_families=allowed_model_families, max_iterations=1)
    with pytest.raises(Exception, match='mock algo init'):
        automl.search()
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
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]['score'] == 1.234
        assert automl.results['pipeline_results'][pipeline_id]['score'] == automl.results['pipeline_results'][pipeline_id]['validation_score']


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
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]['score'] == 1.234
        assert automl.results['pipeline_results'][pipeline_id]['score'] == automl.results['pipeline_results'][pipeline_id]['validation_score']


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
        assert automl.results['pipeline_results'][pipeline_id]['cv_data'][0]['score'] == 1.234
        assert automl.results['pipeline_results'][pipeline_id]['score'] == automl.results['pipeline_results'][pipeline_id]['validation_score']


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
        np.testing.assert_almost_equal(automl.results['pipeline_results'][0]['cv_data'][fold]['score'], 0.0, decimal=4)
    np.testing.assert_almost_equal(automl.results['pipeline_results'][0]['score'], 0.0, decimal=4)
    np.testing.assert_almost_equal(automl.results['pipeline_results'][0]['validation_score'], 0.0, decimal=4)


def test_allowed_pipelines_with_incorrect_problem_type(dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    # checks that not setting allowed_pipelines does not error out
    AutoMLSearch(X_train=X, y_train=y, problem_type='binary')

    with pytest.raises(ValueError, match="is not compatible with problem_type"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='regression', allowed_pipelines=[dummy_binary_pipeline_class])


def test_main_objective_problem_type_mismatch(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(X_train=X, y_train=y, problem_type='binary', objective='R2')


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

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1, allowed_pipelines=[dummy_binary_pipeline_class])
    automl.search()

    mock_score.return_value = {'Log Loss Binary': 0.1234}

    test_pipeline = dummy_binary_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline)

    assert len(automl.rankings) == 2
    assert 0.1234 in automl.rankings['score'].values


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_no_search(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)

    mock_score.return_value = {'Log Loss Binary': 0.1234}
    test_pipeline = dummy_binary_pipeline_class(parameters={})

    automl.add_to_rankings(test_pipeline)
    assert isinstance(automl.data_splitter, StratifiedKFold)
    assert len(automl.rankings) == 1
    assert 0.1234 in automl.rankings['score'].values
    assert np.isnan(automl.results['pipeline_results'][0]['percent_better_than_baseline'])
    assert all(np.isnan(res) for res in automl.results['pipeline_results'][0]['percent_better_than_baseline_all_objectives'].values())
    automl.search()
    assert len(automl.rankings) == 2


@patch('evalml.pipelines.RegressionPipeline.score')
def test_add_to_rankings_regression_large(mock_score, dummy_regression_pipeline_class):
    X = pd.DataFrame({'col_0': [i for i in range(101000)]})
    y = pd.Series([i for i in range(101000)])

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_time=1, max_iterations=1, n_jobs=1)
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    test_pipeline = dummy_regression_pipeline_class(parameters={})
    mock_score.return_value = {automl.objective.name: 0.1234}

    automl.add_to_rankings(test_pipeline)
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert len(automl.rankings) == 1
    assert 0.1234 in automl.rankings['score'].values


@patch('evalml.pipelines.RegressionPipeline.score')
def test_add_to_rankings_regression(mock_score, dummy_regression_pipeline_class, X_y_regression):
    X, y = X_y_regression

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='regression', max_time=1, max_iterations=1, n_jobs=1)
    test_pipeline = dummy_regression_pipeline_class(parameters={})
    mock_score.return_value = {automl.objective.name: 0.1234}

    automl.add_to_rankings(test_pipeline)
    assert isinstance(automl.data_splitter, KFold)
    assert len(automl.rankings) == 1
    assert 0.1234 in automl.rankings['score'].values


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_duplicate(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 0.1234}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1, allowed_pipelines=[dummy_binary_pipeline_class])
    automl.search()

    test_pipeline = dummy_binary_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline)

    test_pipeline_duplicate = dummy_binary_pipeline_class(parameters={})
    assert automl.add_to_rankings(test_pipeline_duplicate) is None


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_add_to_rankings_trained(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1, allowed_pipelines=[dummy_binary_pipeline_class])
    automl.search()

    mock_score.return_value = {'Log Loss Binary': 0.1234}
    test_pipeline = dummy_binary_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline)

    class CoolBinaryClassificationPipeline(dummy_binary_pipeline_class):
        name = "Cool Binary Classification Pipeline"

    mock_fit.return_value = CoolBinaryClassificationPipeline(parameters={})
    test_pipeline_trained = CoolBinaryClassificationPipeline(parameters={}).fit(X, y)
    automl.add_to_rankings(test_pipeline_trained)

    assert list(automl.rankings['score'].values).count(0.1234) == 2


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_has_searched(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)
    mock_score.return_value = {automl.objective.name: 1.0}
    assert not automl.has_searched

    automl.search()
    assert automl.has_searched


def test_no_search(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary')
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)

    df_columns = ["id", "pipeline_name", "score", "validation_score", "percent_better_than_baseline",
                  "high_variance_cv", "parameters"]
    assert (automl.rankings.columns == df_columns).all()
    assert (automl.full_rankings.columns == df_columns).all()

    assert automl._data_check_results is None

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
def test_describe_pipeline(mock_fit, mock_score, caplog, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)
    automl.search()
    out = caplog.text

    assert "Searching up to 1 pipelines. " in out

    assert len(automl.results['pipeline_results']) == 1
    caplog.clear()
    automl.describe_pipeline(0)
    out = caplog.text
    assert "Mode Baseline Binary Classification Pipeline" in out
    assert "Problem Type: binary" in out
    assert "Model Family: Baseline" in out
    assert "* strategy : mode" in out
    assert "Total training time (including CV): " in out
    assert "Log Loss Binary # Training # Validation" in out
    assert "0                      1.000     66.000       34.000" in out
    assert "1                      1.000     67.000       33.000" in out
    assert "2                      1.000     67.000       33.000" in out
    assert "mean                   1.000          -            -" in out
    assert "std                    0.000          -            -" in out
    assert "coef of var            0.000          -            -" in out


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_results_getter(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1)

    assert automl.results == {'pipeline_results': {},
                              'search_order': [],
                              'errors': []}

    mock_score.return_value = {'Log Loss Binary': 1.0}
    automl.search()

    assert automl.results['pipeline_results'][0]['score'] == 1.0

    with pytest.raises(AttributeError, match='set attribute'):
        automl.results = 2.0

    automl.results['pipeline_results'][0]['score'] = 2.0
    assert automl.results['pipeline_results'][0]['score'] == 1.0


@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@pytest.mark.parametrize("target_type", numeric_and_boolean_dtypes + categorical_dtypes + ['Int64', 'boolean'])
def test_targets_data_types_classification(data_type, automl_type, target_type):
    if data_type == 'np' and target_type not in numeric_and_boolean_dtypes + categorical_dtypes:
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
    if target_type in categorical_dtypes:
        if target_type == "category":
            y = pd.Categorical(y)
    elif "int" in target_type.lower():
        y = y.map({unique_vals[i]: int(i) for i in range(len(unique_vals))})
    elif "float" in target_type.lower():
        y = y.map({unique_vals[i]: float(i) for i in range(len(unique_vals))})

    y = y.astype(target_type)

    if data_type == 'np':
        X = X.to_numpy()
        y = y.to_numpy()

    elif data_type == 'ww':
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=automl_type, max_iterations=3, n_jobs=1)
    automl.search()
    for pipeline_id, pipeline_result in automl.results['pipeline_results'].items():
        cv_data = pipeline_result['cv_data']
        for fold in cv_data:
            all_objective_scores = fold["all_objective_scores"]
            for score in all_objective_scores.values():
                assert score is not None

    assert len(automl.full_rankings) == 3
    assert not automl.full_rankings['score'].isnull().values.any()


class KeyboardInterruptOnKthPipeline:
    """Helps us time when the test will send a KeyboardInterrupt Exception to search."""

    def __init__(self, k):
        self.n_calls = 1
        self.k = k

    def __call__(self, pipeline_class, parameters, automl_obj):
        """Raises KeyboardInterrupt on the kth call.

        Arguments are ignored but included to meet the call back API.
        """
        if self.n_calls == self.k:
            self.n_calls += 1
            raise KeyboardInterrupt
        else:
            self.n_calls += 1


# These are used to mock return values to the builtin "input" function.
interrupt = ["y"]
interrupt_after_bad_message = ["No.", "Yes!", "y"]
dont_interrupt = ["n"]
dont_interrupt_after_bad_message = ["Yes", "yes.", "n"]


@pytest.mark.parametrize("when_to_interrupt,user_input,number_results",
                         [(1, interrupt, 0),
                          (1, interrupt_after_bad_message, 0),
                          (1, dont_interrupt, 5),
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
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"F1": 1.0})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_catch_keyboard_interrupt(mock_fit, mock_score, mock_input,
                                  when_to_interrupt, user_input, number_results,
                                  X_y_binary):
    mock_input.side_effect = user_input
    X, y = X_y_binary
    callback = KeyboardInterruptOnKthPipeline(k=when_to_interrupt)
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=5, start_iteration_callback=callback, objective="f1")
    automl.search()
    assert len(automl._results['pipeline_results']) == number_results
    if number_results == 0:
        with pytest.raises(PipelineNotFoundError):
            automl.best_pipeline


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.automl.AutoMLSearch._evaluate_pipelines')
def test_pipelines_in_batch_return_nan(mock_evaluate_pipelines, mock_next_batch, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    mock_evaluate_pipelines.reset_mock()
    mock_next_batch.reset_mock()
    mock_evaluate_pipelines.side_effect = [[0, 0],  # first batch
                                           [0, np.nan],  # second batch
                                           [np.nan, np.nan]]  # third batch, should raise error
    mock_next_batch.side_effect = [[dummy_binary_pipeline_class(parameters={}), dummy_binary_pipeline_class(parameters={})] for i in range(3)]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', allowed_pipelines=[dummy_binary_pipeline_class], n_jobs=1)
    with pytest.raises(AutoMLSearchException, match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"):
        automl.search()


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.automl.AutoMLSearch._evaluate_pipelines')
def test_pipelines_in_batch_return_none(mock_evaluate_pipelines, mock_next_batch, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    mock_evaluate_pipelines.side_effect = [[0, 0],  # first batch
                                           [0, np.nan],  # second batch
                                           [None, None]]  # third batch, should raise error
    mock_next_batch.side_effect = [[dummy_binary_pipeline_class(parameters={}), dummy_binary_pipeline_class(parameters={})] for i in range(3)]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', allowed_pipelines=[dummy_binary_pipeline_class], n_jobs=1)
    with pytest.raises(AutoMLSearchException, match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"):
        automl.search()


@patch('evalml.automl.automl_search.split_data')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_error_during_train_test_split(mock_fit, mock_score, mock_split_data, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 1.0}
    mock_split_data.side_effect = RuntimeError()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', objective='Accuracy Binary', max_iterations=2, optimize_thresholds=True, train_best_pipeline=False)
    automl.search()
    for pipeline in automl.results['pipeline_results'].values():
        assert np.isnan(pipeline['score'])


all_objectives = get_core_objectives("binary") + get_core_objectives("multiclass") + get_core_objectives("regression")


@pytest.mark.parametrize("objective,pipeline_scores,baseline_score,problem_type_value",
                         product(all_objectives + [CostBenefitMatrix],
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
    baseline_pipeline_class = {ProblemTypes.BINARY: "evalml.pipelines.ModeBaselineBinaryPipeline",
                               ProblemTypes.MULTICLASS: "evalml.pipelines.ModeBaselineMulticlassPipeline",
                               ProblemTypes.REGRESSION: "evalml.pipelines.MeanBaselineRegressionPipeline",
                               ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesBaselineRegressionPipeline"
                               }[problem_type_value]

    class DummyPipeline(pipeline_class):
        problem_type = problem_type_value

        def fit(self, *args, **kwargs):
            """Mocking fit"""

    class Pipeline1(DummyPipeline):
        name = "Pipeline1"

    class Pipeline2(DummyPipeline):
        name = "Pipeline2"

    mock_score_1 = MagicMock(return_value={objective.name: pipeline_scores[0]})
    mock_score_2 = MagicMock(return_value={objective.name: pipeline_scores[1]})
    Pipeline1.score = mock_score_1
    Pipeline2.score = mock_score_2

    if objective.name.lower() == "cost benefit matrix":
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type_value, max_iterations=3,
                              allowed_pipelines=[Pipeline1, Pipeline2], objective=objective(0, 0, 0, 0),
                              additional_objectives=[], n_jobs=1)
    elif problem_type_value == ProblemTypes.TIME_SERIES_REGRESSION:
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type_value, max_iterations=3,
                              allowed_pipelines=[Pipeline1, Pipeline2], objective=objective,
                              additional_objectives=[], problem_configuration={'gap': 0, 'max_delay': 0}, n_jobs=1)
    else:
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type_value, max_iterations=3,
                              allowed_pipelines=[Pipeline1, Pipeline2], objective=objective,
                              additional_objectives=[], n_jobs=1)

    with patch(baseline_pipeline_class + ".score", return_value={objective.name: baseline_score}):
        automl.search(data_checks=None)
        scores = dict(zip(automl.rankings.pipeline_name, automl.rankings.percent_better_than_baseline))
        baseline_name = next(name for name in automl.rankings.pipeline_name if name not in {"Pipeline1", "Pipeline2"})
        answers = {"Pipeline1": round(objective.calculate_percent_difference(pipeline_scores[0], baseline_score), 2),
                   "Pipeline2": round(objective.calculate_percent_difference(pipeline_scores[1], baseline_score), 2),
                   baseline_name: round(objective.calculate_percent_difference(baseline_score, baseline_score), 2)}
        for name in answers:
            np.testing.assert_almost_equal(scores[name], answers[name], decimal=3)


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression", "time series regression"])
@patch("evalml.pipelines.ModeBaselineBinaryPipeline.fit")
@patch("evalml.pipelines.ModeBaselineMulticlassPipeline.fit")
@patch("evalml.pipelines.MeanBaselineRegressionPipeline.fit")
@patch("evalml.pipelines.TimeSeriesBaselineRegressionPipeline.fit")
def test_percent_better_than_baseline_computed_for_all_objectives(mock_time_series_baseline_regression_fit,
                                                                  mock_baseline_regression_fit,
                                                                  mock_baseline_multiclass_fit,
                                                                  mock_baseline_binary_fit,
                                                                  problem_type,
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
    baseline_pipeline_class = {"binary": "evalml.pipelines.ModeBaselineBinaryPipeline",
                               "multiclass": "evalml.pipelines.ModeBaselineMulticlassPipeline",
                               "regression": "evalml.pipelines.MeanBaselineRegressionPipeline",
                               "time series regression": "evalml.pipelines.TimeSeriesBaselineRegressionPipeline"
                               }[problem_type]

    class DummyPipeline(pipeline_class):
        name = "Dummy 1"
        problem_type = problem_type_enum

        def fit(self, *args, **kwargs):
            """Mocking fit"""

    core_objectives = get_core_objectives(problem_type)
    mock_scores = {obj.name: i for i, obj in enumerate(core_objectives)}
    mock_baseline_scores = {obj.name: i + 1 for i, obj in enumerate(core_objectives)}
    answer = {obj.name: obj.calculate_percent_difference(mock_scores[obj.name],
                                                         mock_baseline_scores[obj.name]) for obj in core_objectives}

    mock_score_1 = MagicMock(return_value=mock_scores)
    DummyPipeline.score = mock_score_1

    # specifying problem_configuration for all problem types for conciseness
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type, max_iterations=2,
                          allowed_pipelines=[DummyPipeline], objective="auto", problem_configuration={'gap': 1, 'max_delay': 1})

    with patch(baseline_pipeline_class + ".score", return_value=mock_baseline_scores):
        automl.search(data_checks=None)
        assert len(automl.results['pipeline_results']) == 2, "This tests assumes only one non-baseline pipeline was run!"
        pipeline_results = automl.results['pipeline_results'][1]
        assert pipeline_results["percent_better_than_baseline_all_objectives"] == answer
        assert pipeline_results['percent_better_than_baseline'] == pipeline_results["percent_better_than_baseline_all_objectives"][automl.objective.name]


@pytest.mark.parametrize("fold_scores", [[2, 4, 6], [np.nan, 4, 6]])
@patch("evalml.pipelines.ModeBaselineBinaryPipeline.score", return_value={'Log Loss Binary': 1, 'F1': 1})
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_percent_better_than_baseline_scores_different_folds(mock_fit,
                                                             mock_score,
                                                             mock_baseline_score,
                                                             fold_scores,
                                                             dummy_binary_pipeline_class,
                                                             X_y_binary):
    # Test that percent-better-than-baseline is correctly computed when scores differ across folds
    X, y = X_y_binary

    mock_score.side_effect = [{"Log Loss Binary": 1, "F1": val} for val in fold_scores]

    class DummyPipeline(dummy_binary_pipeline_class):
        name = "Dummy 1"
        problem_type = ProblemTypes.BINARY

    f1 = get_objective("f1")()

    if np.isnan(fold_scores[0]):
        answer = np.nan
    else:
        answer = f1.calculate_percent_difference(4, 1)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=2,
                          allowed_pipelines=[DummyPipeline], objective="log loss binary", additional_objectives=["f1"])

    automl.search(data_checks=None)
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
    automl.search(data_checks=None)
    # every nth batch a stacked ensemble will be trained
    if max_iterations is None:
        max_iterations = 5  # Default value for max_iterations

    pipeline_names = automl.rankings['pipeline_name']
    if max_iterations < _get_first_stacked_classifier_no():
        assert not pipeline_names.str.contains('Ensemble').any()
    elif use_ensembling:
        assert pipeline_names.str.contains('Ensemble').any()
        assert f"Ensembling will run at the {_get_first_stacked_classifier_no()} iteration" in caplog.text

    else:
        assert not pipeline_names.str.contains('Ensemble').any()


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

    automl.search(data_checks=None)
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


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_one_allowed_pipeline_ensembling_disabled(mock_pipeline_fit, mock_score, X_y_binary, logistic_regression_binary_pipeline_class, caplog):
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.RANDOM_FOREST]) + 1
    # Checks that when len(allowed_pipeline) == 1, ensembling is not run, even if set to True
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations, allowed_model_families=[ModelFamily.RANDOM_FOREST], ensembling=True)
    automl.search(data_checks=None)
    assert "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run." in caplog.text

    pipeline_names = automl.rankings['pipeline_name']
    assert not pipeline_names.str.contains('Ensemble').any()

    caplog.clear()
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.LINEAR_MODEL]) + 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations, allowed_pipelines=[logistic_regression_binary_pipeline_class], ensembling=True)
    automl.search(data_checks=None)
    pipeline_names = automl.rankings['pipeline_name']
    assert not pipeline_names.str.contains('Ensemble').any()
    assert "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run." in caplog.text

    # Check that ensembling runs when len(allowed_model_families) == 1 but len(allowed_pipelines) > 1
    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations, allowed_model_families=[ModelFamily.LINEAR_MODEL], ensembling=True)
    automl.search(data_checks=None)
    pipeline_names = automl.rankings['pipeline_name']
    assert pipeline_names.str.contains('Ensemble').any()
    assert "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run." not in caplog.text


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_max_iterations_less_than_ensembling_disabled(mock_pipeline_fit, mock_score, X_y_binary, caplog):
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.LINEAR_MODEL])
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=max_iterations - 1, allowed_model_families=[ModelFamily.LINEAR_MODEL], ensembling=True)
    automl.search(data_checks=None)
    assert f"Ensembling is set to True, but max_iterations is too small, so ensembling will not run. Set max_iterations >= {max_iterations} to run ensembling." in caplog.text

    pipeline_names = automl.rankings['pipeline_name']
    assert not pipeline_names.str.contains('Ensemble').any()


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_max_batches_less_than_ensembling_disabled(mock_pipeline_fit, mock_score, X_y_binary, caplog):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_batches=2, allowed_model_families=[ModelFamily.LINEAR_MODEL], ensembling=True)
    automl.search(data_checks=None)
    first_ensemble_batch = 1 + len(automl.allowed_pipelines) + 1  # First batch + each pipeline batch
    assert f"Ensembling is set to True, but max_batches is too small, so ensembling will not run. Set max_batches >= {first_ensemble_batch} to run ensembling." in caplog.text

    pipeline_names = automl.rankings['pipeline_name']
    assert not pipeline_names.str.contains('Ensemble').any()


@pytest.mark.parametrize("max_batches", [1, 2, 5, 10])
@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_max_batches_output(mock_pipeline_fit, mock_score, max_batches, X_y_binary, caplog):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=None, max_batches=max_batches)
    automl.search(data_checks=None)

    output = caplog.text
    for batch_number in range(1, max_batches + 1):
        if batch_number == 1:
            correct_output = len(automl.allowed_pipelines) + 1
        else:
            correct_output = automl._pipelines_per_batch
        assert output.count(f"Batch {batch_number}: ") == correct_output


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_max_batches_plays_nice_with_other_stopping_criteria(mock_fit, mock_score, X_y_binary):
    X, y = X_y_binary

    # Use the old default when all are None
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", objective="Log Loss Binary")
    automl.search(data_checks=None)
    assert len(automl.results["pipeline_results"]) == len(get_estimators(problem_type='binary')) + 1

    # Use max_iterations when both max_iterations and max_batches are set
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", objective="Log Loss Binary", max_batches=10,
                          max_iterations=6)
    automl.search(data_checks=None)
    assert len(automl.results["pipeline_results"]) == 6

    # Don't change max_iterations when only max_iterations is set
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=4)
    automl.search(data_checks=None)
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
    X, y = X_y_binary
    y[:] = 0
    y[0] = 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", n_jobs=1)
    with pytest.raises(Exception, match="Missing target values in the"):
        with pytest.warns(UserWarning):
            automl.search(data_checks="disabled")

    y[1] = 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", n_jobs=1)
    with pytest.raises(Exception, match="Missing target values in the"):
        with pytest.warns(UserWarning):
            automl.search(data_checks="disabled")

    y[2] = 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", n_jobs=1)
    automl.search(data_checks="disabled")


@patch('evalml.pipelines.MulticlassClassificationPipeline.score')
@patch('evalml.pipelines.MulticlassClassificationPipeline.fit')
def test_data_splitter_multi(mock_fit, mock_score, X_y_multi):
    X, y = X_y_multi
    y[:] = 1
    y[0] = 0

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            automl.search(data_checks="disabled")

    y[1] = 2
    # match based on regex, since data split doesn't have a random seed for reproducibility
    # regex matches the set {} and expects either 2 sets (missing in both train and test)
    #   or 1 set of multiple elements (both missing in train or both in test)
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    with pytest.raises(Exception, match=r"(\{\d?\}.+\{\d?\})|(\{.+\,.+\})"):
        with pytest.warns(UserWarning):
            automl.search(data_checks="disabled")

    y[1] = 0
    y[2:4] = 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            automl.search(data_checks="disabled")

    y[4] = 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            automl.search(data_checks="disabled")

    y[5] = 0
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', n_jobs=1)
    automl.search(data_checks="disabled")


@patch('evalml.tuners.skopt_tuner.SKOptTuner.add')
def test_iterative_algorithm_pipeline_hyperparameters_make_pipeline_other_errors(mock_add, X_y_multi):
    X, y = X_y_multi
    custom_hyperparameters = {
        "Imputer": {
            "numeric_impute_strategy": ["most_frequent", "mean"]
        }
    }
    estimators = get_estimators('multiclass', [ModelFamily.EXTRA_TREES])

    pipelines = [make_pipeline(X, y, estimator, 'multiclass', custom_hyperparameters) for estimator in estimators]
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

    invalid_pipelines = [make_pipeline(X, y, estimator, 'multiclass', invalid_custom_hyperparameters) for estimator in estimators]
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', allowed_pipelines=invalid_pipelines)
    with pytest.raises(ValueError, match="Default parameters for components"):
        automl.search()

    invalid_pipelines = [make_pipeline(X, y, estimator, 'multiclass', larger_invalid) for estimator in estimators]
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
    pipelines = [make_pipeline(X, y, estimator, 'multiclass', custom_hyperparameters) for estimator in estimators]

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='multiclass', allowed_pipelines=pipelines)
    automl.search()
    assert automl.best_pipeline.hyperparameters['Imputer']['numeric_impute_strategy'] == ["mean"]

    invalid_pipelines = [make_pipeline(X, y, estimator, 'multiclass', larger_custom) for estimator in estimators]
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

        def __init__(self, n_jobs=-1, random_state=0):
            super().__init__(parameters={"n_jobs": n_jobs}, component_obj=None, random_state=random_state)

    class Pipeline1(BinaryClassificationPipeline):
        name = "Pipeline 1"
        component_graph = [MockEstimatorWithNJobs]

    class Pipeline2(BinaryClassificationPipeline):
        name = "Pipeline 2"
        component_graph = [MockEstimatorWithNJobs]

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', n_jobs=3, max_batches=2,
                          allowed_pipelines=[Pipeline1, Pipeline2, dummy_binary_pipeline_class])
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
    automl.search(data_checks='disabled')  # DataChecks disabled since the data is small
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
def test_automl_respects_random_state(mock_fit, mock_score, X_y_binary, dummy_classifier_estimator_class):

    expected_random_state = get_random_state(42)
    X, y = X_y_binary

    class DummyPipeline(BinaryClassificationPipeline):
        component_graph = [dummy_classifier_estimator_class]
        num_pipelines_different_seed = 0
        num_pipelines_init = 0

        def __init__(self, parameters, random_state):
            random_state = get_random_state(random_state)
            is_diff_random_state = not check_random_state_equality(random_state, expected_random_state)
            self.__class__.num_pipelines_init += 1
            self.__class__.num_pipelines_different_seed += is_diff_random_state
            super().__init__(parameters, random_state)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", allowed_pipelines=[DummyPipeline],
                          random_state=expected_random_state, max_iterations=10)
    automl.search()
    assert DummyPipeline.num_pipelines_different_seed == 0 and DummyPipeline.num_pipelines_init


@patch('evalml.pipelines.BinaryClassificationPipeline.score', return_value={"Log Loss Binary": 0.8})
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_error_callback(mock_fit, mock_score, X_y_binary, caplog):
    X, y = X_y_binary
    msg = 'all your model are belong to us'
    mock_fit.side_effect = Exception(msg)
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", error_callback=None, train_best_pipeline=False, n_jobs=1)
    automl.search()
    assert msg in caplog.text

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", error_callback=silent_error_callback, train_best_pipeline=False, n_jobs=1)
    automl.search()
    assert msg not in caplog.text

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", error_callback=log_error_callback, train_best_pipeline=False, n_jobs=1)
    automl.search()
    assert msg in caplog.text

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", error_callback=raise_error_callback, train_best_pipeline=False, n_jobs=1)
    with pytest.raises(Exception, match="all your model are belong to us"):
        automl.search()
    assert "AutoMLSearch raised a fatal exception: all your model are belong to us" in caplog.text
    assert "fit" in caplog.text  # Check stack trace logged

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", error_callback=log_and_save_error_callback, train_best_pipeline=False, n_jobs=1)
    automl.search()
    assert "AutoML search encountered an exception: all your model are belong to us" in caplog.text
    assert "fit" in caplog.text  # Check stack trace logged
    # first automl batch, times 3 for 3-fold cross validation
    assert len(automl._results['errors']) == (1 + len(get_estimators(problem_type='binary'))) * 3
    for e in automl._results['errors']:
        assert str(e) == msg

    caplog.clear()
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", error_callback=raise_and_save_error_callback, train_best_pipeline=False, n_jobs=1)
    with pytest.raises(Exception, match="all your model are belong to us"):
        automl.search()
    assert "AutoMLSearch raised a fatal exception: all your model are belong to us" in caplog.text
    assert "fit" in caplog.text  # Check stack trace logged
    assert len(automl._results['errors']) == 1  # Raises exception at first error
    for e in automl._results['errors']:
        assert str(e) == msg


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
    msg = "user_parameters must be a dict containing values for at least the gap and max_delay parameters"
    with pytest.raises(ValueError, match=msg):
        AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression")
    with pytest.raises(ValueError, match=msg):
        AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression", problem_configuration={"gap": 3})

    problem_config = AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression",
                                  problem_configuration={"max_delay": 2, "gap": 3}).problem_configuration
    assert problem_config == {"max_delay": 2, "gap": 3}


@patch('evalml.pipelines.TimeSeriesRegressionPipeline.score', return_value={"R2": 0.3})
@patch('evalml.pipelines.TimeSeriesRegressionPipeline.fit')
def test_automl_time_series_regression(mock_fit, mock_score, X_y_regression):
    X, y = X_y_regression

    configuration = {"gap": 0, "max_delay": 0, 'delay_target': False, 'delay_features': True}

    class Pipeline1(TimeSeriesRegressionPipeline):
        name = "Pipeline 1"
        component_graph = ["Delayed Feature Transformer", "Random Forest Regressor"]

    class Pipeline2(TimeSeriesRegressionPipeline):
        name = "Pipeline 2"
        component_graph = ["Delayed Feature Transformer", "Elastic Net Regressor"]

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression", problem_configuration=configuration,
                          allowed_pipelines=[Pipeline1, Pipeline2], max_batches=2)
    automl.search()
    assert isinstance(automl.data_splitter, TimeSeriesSplit)
    for result in automl.results['pipeline_results'].values():
        if result["id"] == 0:
            continue
        assert result['parameters']['Delayed Feature Transformer'] == configuration
        assert result['parameters']['pipeline'] == configuration


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
    assert automl.best_pipeline.threshold == 0.5

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
    random_state = [0, 0, 1]
    for state in random_state:
        a = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type, random_state=state, max_iterations=1)
        a.search()
        data_splitters.append([[set(train), set(test)] for train, test in a.data_splitter.split(X, y)])
    # append split from last random state again, should be referencing same datasplit object
    data_splitters.append([[set(train), set(test)] for train, test in a.data_splitter.split(X, y)])

    assert data_splitters[0] == data_splitters[1]
    assert data_splitters[1] != data_splitters[2]
    assert data_splitters[2] == data_splitters[3]
