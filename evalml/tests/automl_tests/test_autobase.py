from unittest.mock import patch

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from evalml import AutoClassificationSearch, AutoRegressionSearch
from evalml.pipelines import LogisticRegressionBinaryPipeline
from evalml.tuners import RandomSearchTuner


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
    assert "No search limit is set. Set using max_time or max_pipelines." in out


def test_search_order(X_y):
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=3)
    automl.search(X, y)
    correct_order = [0, 1, 2]
    assert automl.results['search_order'] == correct_order


def test_transform_parameters():
    automl = AutoClassificationSearch(max_pipelines=1, random_state=100, n_jobs=6)
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'most_frequent'
        },
        'Logistic Regression Classifier': {
            'penalty': 'l2',
            'C': 8.444214828324364
        }
    }
    parameters_dict = {
        'Simple Imputer': {'impute_strategy': 'most_frequent'},
        'Logistic Regression Classifier': {'penalty': 'l2', 'C': 8.444214828324364, 'n_jobs': 6}
    }
    assert automl._transform_parameters(LogisticRegressionBinaryPipeline, parameters, 0) == parameters_dict


@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_pipeline_fit_raises(mock_fit, X_y):
    msg = 'all your model are belong to us'
    mock_fit.side_effect = Exception(msg)
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=1)
    with pytest.raises(Exception, match=msg):
        automl.search(X, y)

    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y, raise_errors=False)
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0].get('cv_data', {})
    for cv_scores in cv_scores_all:
        for name, score in cv_scores['all_objective_scores'].items():
            if name in ['# Training', '# Testing']:
                assert score > 0
            else:
                assert np.isnan(score)


@patch('evalml.objectives.AUC.score')
def test_pipeline_score_raises(mock_score, X_y):
    msg = 'all your model are belong to us'
    mock_score.side_effect = Exception(msg)
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y)
    pipeline_results = automl.results.get('pipeline_results', {})
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0].get('cv_data', {})
    scores = cv_scores_all[0]['all_objective_scores']
    auc_score = scores.pop('AUC')
    assert np.isnan(auc_score)
    assert not np.isnan(list(cv_scores_all[0]['all_objective_scores'].values())).any()

    automl = AutoClassificationSearch(max_pipelines=1)
    automl.search(X, y, raise_errors=False)
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
    automl = AutoClassificationSearch(allowed_model_families=model_families, max_pipelines=2)
    automl.search(X, y)
    assert len(automl.full_rankings) == 2
    assert len(automl.rankings) == 1

    X, y = X_y_reg
    automl = AutoRegressionSearch(allowed_model_families=model_families, max_pipelines=2)
    automl.search(X, y)
    assert len(automl.full_rankings) == 2
    assert len(automl.rankings) == 1


@patch('evalml.pipelines.PipelineBase.fit')
@patch('evalml.guardrails.detect_label_leakage')
def test_detect_label_leakage(mock_detect_label_leakage, mock_fit, capsys, caplog, X_y):
    X, y = X_y
    mock_detect_label_leakage.return_value = {'var 1': 0.1234, 'var 2': 0.5678}
    automl = AutoClassificationSearch(max_pipelines=1, random_state=0)
    automl.search(X, y, raise_errors=False)
    out = caplog.text
    assert "WARNING: Possible label leakage: var 1, var 2" in out


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
        'tuner': RandomSearchTuner,
        'detect_label_leakage': False,
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
        'Possible Pipelines': ['Random Forest Binary Classification Pipeline', 'Logistic Regression Binary Pipeline'],
        'Patience': search_params['patience'],
        'Tolerance': search_params['tolerance'],
        'Cross Validation': 'StratifiedKFold(n_splits=5, random_state=None, shuffle=False)',
        'Tuner': 'RandomSearchTuner',
        'Detect Label Leakage': search_params['detect_label_leakage'],
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


def test_automl_str_no_param_search():
    automl = AutoClassificationSearch()

    param_str_reps = {
        'Objective': 'Log Loss Binary',
        'Max Time': 'None',
        'Max Pipelines': 'None',
        'Possible Pipelines': [
            'Logistic Regression Binary Pipeline',
            'Random Forest Binary Classification Pipeline'],
        'Patience': 'None',
        'Tolerance': '0.0',
        'Cross Validation': 'StratifiedKFold(n_splits=3, random_state=0, shuffle=True)',
        'Tuner': 'SKOptTuner',
        'Detect Label Leakage': 'True',
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
    assert "Possible Pipelines" in str_rep
    assert "Search Results" not in str_rep
