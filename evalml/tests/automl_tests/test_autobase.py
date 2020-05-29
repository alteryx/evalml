from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from evalml import AutoClassificationSearch, AutoRegressionSearch
from evalml.data_checks import (
    DataCheck,
    DataCheckError,
    DataChecks,
    DataCheckWarning,
    EmptyDataChecks
)
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline, get_pipelines
from evalml.problem_types import ProblemTypes
from evalml.tuners import NoParamsException, RandomSearchTuner


def test_pipeline_limits(caplog, X_y):
    X, y = X_y

    automl = AutoClassificationSearch(multiclass=False, max_pipelines=1)
    automl.search(X, y)
    out = caplog.text
    assert "Searching up to 1 pipelines. " in out

    caplog.clear()
    automl = AutoClassificationSearch(multiclass=False, max_time=1)
    automl.search(X, y)
    out = caplog.text
    assert "Will stop searching for new pipelines after 1 seconds" in out

    caplog.clear()
    automl = AutoClassificationSearch(multiclass=False, max_time=1, max_pipelines=5)
    automl.search(X, y)
    out = caplog.text
    assert "Searching up to 5 pipelines. " in out
    assert "Will stop searching for new pipelines after 1 seconds" in out

    caplog.clear()
    automl = AutoClassificationSearch(multiclass=False)
    automl.search(X, y)
    out = caplog.text
    assert "Using default limit of max_pipelines=5." in out


def test_search_order(X_y):
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=3)
    automl.search(X, y)
    correct_order = [0, 1, 2]
    assert automl.results['search_order'] == correct_order


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_pipeline_fit_raises(mock_fit, X_y, caplog):
    msg = 'all your model are belong to us'
    mock_fit.side_effect = Exception(msg)
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=1)
    with pytest.raises(Exception, match=msg):
        automl.search(X, y, raise_errors=True)
    out = caplog.text
    assert 'Exception during automl search' in out

    caplog.clear()
    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y, raise_errors=False)
    out = caplog.text
    assert 'Exception during automl search' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1

    cv_scores_all = pipeline_results[0].get('cv_data', {})
    for cv_scores in cv_scores_all:
        for name, score in cv_scores['all_objective_scores'].items():
            if name in ['# Training', '# Testing']:
                assert score > 0
            else:
                assert np.isnan(score)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
def test_pipeline_score_raises(mock_score, X_y, caplog):
    msg = 'all your model are belong to us'
    mock_score.side_effect = Exception(msg)
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=1)
    with pytest.raises(Exception, match=msg):
        automl.search(X, y, raise_errors=True)
    out = caplog.text
    assert 'Exception during automl search' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 0

    caplog.clear()
    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y, raise_errors=False)
    out = caplog.text
    assert 'Exception during automl search' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0].get('cv_data', {})
    assert np.isnan(list(cv_scores_all[0]['all_objective_scores'].values())).any()


@patch('evalml.objectives.AUC.score')
def test_objective_score_raises(mock_score, X_y, caplog):
    msg = 'all your model are belong to us'
    mock_score.side_effect = Exception(msg)
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y, raise_errors=True)
    out = caplog.text
    assert 'Error in PipelineBase.score while scoring objective AUC: all your model are belong to us' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0].get('cv_data', {})
    scores = cv_scores_all[0]['all_objective_scores']
    auc_score = scores.pop('AUC')
    assert np.isnan(auc_score)
    assert not np.isnan(list(cv_scores_all[0]['all_objective_scores'].values())).any()

    caplog.clear()
    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y, raise_errors=False)
    out = caplog.text
    assert 'Error in PipelineBase.score while scoring objective AUC: all your model are belong to us' in out
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0].get('cv_data', {})
    scores = cv_scores_all[0]['all_objective_scores']
    auc_score = scores.pop('AUC')
    assert np.isnan(auc_score)
    assert not np.isnan(list(cv_scores_all[0]['all_objective_scores'].values())).any()


def test_rankings(X_y, X_y_reg):
    X, y = X_y
    model_families = ['random_forest']
    automl = AutoClassificationSearch(allowed_model_families=model_families, max_pipelines=3)
    automl.search(X, y)
    assert len(automl.full_rankings) == 3
    assert len(automl.rankings) == 2

    X, y = X_y_reg
    automl = AutoRegressionSearch(allowed_model_families=model_families, max_pipelines=3)
    automl.search(X, y)
    assert len(automl.full_rankings) == 3
    assert len(automl.rankings) == 2


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_str_search(mock_fit, X_y):
    def _dummy_callback(param1, param2):
        return None

    X, y = X_y
    search_params = {
        'objective': 'F1',
        'max_time': 100,
        'max_pipelines': 5,
        'patience': 2,
        'tolerance': 0.5,
        'allowed_model_families': ['random_forest', 'linear_model'],
        'cv': StratifiedKFold(5),
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
        'Max Pipelines': search_params['max_pipelines'],
        'Allowed Pipelines': ['Random Forest Binary Classification Pipeline', 'Logistic Regression Binary Pipeline'],
        'Patience': search_params['patience'],
        'Tolerance': search_params['tolerance'],
        'Cross Validation': 'StratifiedKFold(n_splits=5, random_state=None, shuffle=False)',
        'Tuner': 'RandomSearchTuner',
        'Start Iteration Callback': '_dummy_callback',
        'Add Result Callback': None,
        'Additional Objectives': search_params['additional_objectives'],
        'Random State': 'RandomState(MT19937)',
        'n_jobs': search_params['n_jobs'],
        'Optimize Thresholds': search_params['optimize_thresholds']
    }

    automl = AutoClassificationSearch(**search_params)
    str_rep = str(automl)

    for param, value in param_str_reps.items():
        if isinstance(value, list):
            assert f"{param}" in str_rep
            for item in value:
                assert f"\t{str(item)}" in str_rep
        else:
            assert f"{param}: {str(value)}" in str_rep
    assert "Search Results" not in str_rep

    automl.search(X, y, raise_errors=False)
    str_rep = str(automl)
    assert "Search Results:" in str_rep
    assert str(automl.rankings.drop(['parameters'], axis='columns')) in str_rep


def test_automl_data_check_results_is_none_before_search():
    automl = AutoClassificationSearch(max_pipelines=1)
    assert automl.data_check_results is None


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_empty_data_checks(mock_fit, mock_score, X_y):
    X, y = X_y
    mock_score.return_value = {'Log Loss Binary': 1.0}
    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y, data_checks=EmptyDataChecks())
    assert automl.data_check_results is None
    mock_fit.assert_called()
    mock_score.assert_called()


@patch('evalml.data_checks.DefaultDataChecks.validate')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_default_data_checks(mock_fit, mock_score, mock_validate, X_y, caplog):
    X, y = X_y
    mock_score.return_value = {'Log Loss Binary': 1.0}
    mock_validate.return_value = [DataCheckWarning("default data check warning", "DefaultDataChecks")]
    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y)
    out = caplog.text
    assert "default data check warning" in out
    assert automl.data_check_results == mock_validate.return_value
    mock_fit.assert_called()
    mock_score.assert_called()
    mock_validate.assert_called()


def test_automl_data_checks_raises_error(caplog):
    X = pd.DataFrame()
    y = pd.Series()

    class MockDataCheckErrorAndWarning(DataCheck):
        def validate(self, X, y):
            return [DataCheckError("error one", self.name), DataCheckWarning("warning one", self.name)]

    data_checks = DataChecks(data_checks=[MockDataCheckErrorAndWarning()])
    automl = AutoClassificationSearch(max_pipelines=1)

    with pytest.raises(ValueError, match="Data checks raised"):
        automl.search(X, y, data_checks=data_checks)
    out = caplog.text
    assert "error one" in out
    assert "warning one" in out
    assert automl.data_check_results == data_checks.validate(X, y)


def test_automl_not_data_check_object():
    X = pd.DataFrame()
    y = pd.Series()
    automl = AutoClassificationSearch(max_pipelines=1)
    with pytest.raises(ValueError, match="data_checks parameter must be a DataChecks object!"):
        automl.search(X, y, data_checks=1)


def test_automl_str_no_param_search():
    automl = AutoClassificationSearch()

    param_str_reps = {
        'Objective': 'Log Loss Binary',
        'Max Time': 'None',
        'Max Pipelines': 'None',
        'Allowed Pipelines': [
            'Logistic Regression Binary Pipeline',
            'Random Forest Binary Classification Pipeline'],
        'Patience': 'None',
        'Tolerance': '0.0',
        'Cross Validation': 'StratifiedKFold(n_splits=3, random_state=0, shuffle=True)',
        'Tuner': 'SKOptTuner',
        'Additional Objectives': [
            'Accuracy Binary',
            'Balanced Accuracy Binary',
            'F1',
            'Precision',
            'AUC',
            'MCC Binary'],
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
@patch('evalml.automl.auto_search_base.get_pipelines')
def test_automl_feature_selection(mock_get_pipelines, mock_fit, mock_score, X_y):
    X, y = X_y
    mock_score.return_value = {'Log Loss Binary': 1.0}

    class MockFeatureSelectionPipeline(BinaryClassificationPipeline):
        component_graph = ['RF Classifier Select From Model', 'Logistic Regression Classifier']

        def fit(self, X, y):
            """Mock fit, noop"""

    allowed_pipelines = [MockFeatureSelectionPipeline]
    mock_get_pipelines.return_value = allowed_pipelines
    start_iteration_callback = MagicMock()
    automl = AutoClassificationSearch(max_pipelines=2, start_iteration_callback=start_iteration_callback)
    automl.search(X, y)

    assert start_iteration_callback.call_count == 2
    proposed_parameters = start_iteration_callback.call_args_list[1][0][1]
    assert proposed_parameters.keys() == {'RF Classifier Select From Model', 'Logistic Regression Classifier'}
    assert proposed_parameters['RF Classifier Select From Model']['number_features'] == X.shape[1]


@patch('evalml.tuners.random_search_tuner.RandomSearchTuner.is_search_space_exhausted')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_tuner_exception(mock_fit, mock_score, mock_is_search_space_exhausted, X_y):
    mock_score.return_value = {'Log Loss Binary': 1.0}
    X, y = X_y
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    mock_is_search_space_exhausted.side_effect = NoParamsException(error_text)
    clf = AutoRegressionSearch(objective="R2", tuner_class=RandomSearchTuner, max_pipelines=10)
    with pytest.raises(NoParamsException, match=error_text):
        clf.search(X, y)


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_automl_algorithm(mock_fit, mock_score, mock_algo_next_batch, X_y):
    X, y = X_y
    mock_score.return_value = {'Log Loss Binary': 1.0}
    mock_algo_next_batch.side_effect = StopIteration("that's all, folks")
    automl = AutoClassificationSearch(max_pipelines=5)
    automl.search(X, y)
    assert automl.data_check_results is None
    mock_fit.assert_called()
    mock_score.assert_called()
    assert mock_algo_next_batch.call_count == 1
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    assert pipeline_results[0].get('score') == 1.0


@patch('evalml.automl.automl_algorithm.IterativeAlgorithm.__init__')
def test_automl_allowed_pipelines_algorithm(mock_algo_init, dummy_binary_pipeline_class, X_y):
    mock_algo_init.side_effect = Exception('mock algo init')
    X, y = X_y

    allowed_pipelines = [dummy_binary_pipeline_class]
    automl = AutoClassificationSearch(allowed_pipelines=allowed_pipelines, max_pipelines=10)
    with pytest.raises(Exception, match='mock algo init'):
        automl.search(X, y)
    assert mock_algo_init.call_count == 1
    _, kwargs = mock_algo_init.call_args
    assert kwargs['max_pipelines'] == 10
    assert kwargs['allowed_pipelines'] == allowed_pipelines

    allowed_model_families = [ModelFamily.RANDOM_FOREST]
    automl = AutoClassificationSearch(allowed_model_families=allowed_model_families, max_pipelines=1)
    with pytest.raises(Exception, match='mock algo init'):
        automl.search(X, y)
    assert mock_algo_init.call_count == 2
    _, kwargs = mock_algo_init.call_args
    assert kwargs['max_pipelines'] == 1
    assert kwargs['allowed_pipelines'] == get_pipelines(problem_type=ProblemTypes.BINARY, model_families=allowed_model_families)
