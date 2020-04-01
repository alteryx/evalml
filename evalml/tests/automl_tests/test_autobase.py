from unittest.mock import patch

import numpy as np
import pytest

from evalml import AutoClassificationSearch
from evalml.pipelines import LogisticRegressionBinaryPipeline


def test_pipeline_limits(capsys, X_y):
    X, y = X_y

    automl = AutoClassificationSearch(multiclass=False, max_pipelines=1)
    automl.search(X, y, raise_errors=True)
    out, err = capsys.readouterr()
    assert "Searching up to 1 pipelines. " in out

    automl = AutoClassificationSearch(multiclass=False, max_time=1)
    automl.search(X, y, raise_errors=True)
    out, err = capsys.readouterr()
    assert "Will stop searching for new pipelines after 1 seconds" in out

    automl = AutoClassificationSearch(multiclass=False, max_time=1, max_pipelines=5)
    automl.search(X, y, raise_errors=True)
    out, err = capsys.readouterr()
    assert "Searching up to 5 pipelines. " in out
    assert "Will stop searching for new pipelines after 1 seconds" in out

    automl = AutoClassificationSearch(multiclass=False)
    automl.search(X, y, raise_errors=True)
    out, err = capsys.readouterr()
    assert "No search limit is set. Set using max_time or max_pipelines." in out


def test_search_order(X_y):
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=3)
    automl.search(X, y)
    correct_order = [0, 1, 2]
    assert automl.results['search_order'] == correct_order


def test_transform_parameters():
    automl = AutoClassificationSearch(max_pipelines=1, random_state=100, n_jobs=6)
    parameters = [('penalty', 'l2'), ('C', 8.444214828324364), ('impute_strategy', 'most_frequent')]
    parameters_dict = {
        'Simple Imputer': {'impute_strategy': 'most_frequent'},
        'One Hot Encoder': {},
        'Standard Scaler': {},
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
        automl.search(X, y, raise_errors=True)

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
    assert automl._transform_parameters(LogisticRegressionPipeline, parameters, 0) == parameters_dict


def test_autobase_repr():
    automl = AutoClassificationSearch(max_pipelines=20, max_time=20, n_jobs=1)
    automl_repr = repr(automl)
    assert "Max Time: 20" in automl_repr
    assert "Max Pipelines: 20" in automl_repr
    assert "n_jobs: 1" in automl_repr

    automl = AutoRegressionSearch(objective='r2', max_time="200 minutes", tuner=GridSearchTuner)
    automl_repr = repr(automl)
    assert "Max Time: 12000.0" in automl_repr
    assert "Objective: R2" in automl_repr
    assert "Tuner: <class 'evalml.tuners.grid_search_tuner.GridSearchTuner'>" in automl_repr
