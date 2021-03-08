from unittest.mock import patch

import numpy as np
import pandas as pd

from evalml.automl.automl_search import AutoMLSearch
from evalml.automl.engine import EngineBase
from evalml.objectives import F1, LogLossBinary


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_train_and_score_pipelines(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 0.42}
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_batches=1,
                          allowed_pipelines=[dummy_binary_pipeline_class])
    pipeline = dummy_binary_pipeline_class({})
    evaluation_result = EngineBase.train_and_score_pipeline(pipeline, automl, automl.X_train, automl.y_train)
    assert mock_fit.call_count == automl.data_splitter.get_n_splits()
    assert mock_score.call_count == automl.data_splitter.get_n_splits()
    assert evaluation_result.get('training_time') is not None
    assert evaluation_result.get('cv_score_mean') == 0.42
    pd.testing.assert_series_equal(evaluation_result.get('cv_scores'), pd.Series([0.42] * 3))
    for i in range(automl.data_splitter.get_n_splits()):
        assert evaluation_result['cv_data'][i]['all_objective_scores']['Log Loss Binary'] == 0.42


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_train_and_score_pipelines_error(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary, caplog):
    X, y = X_y_binary
    mock_score.side_effect = Exception('yeet')
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_batches=1,
                          allowed_pipelines=[dummy_binary_pipeline_class])
    pipeline = dummy_binary_pipeline_class({})
    evaluation_result = EngineBase.train_and_score_pipeline(pipeline, automl, automl.X_train, automl.y_train)
    assert mock_fit.call_count == automl.data_splitter.get_n_splits()
    assert mock_score.call_count == automl.data_splitter.get_n_splits()
    assert evaluation_result.get('training_time') is not None
    assert np.isnan(evaluation_result.get('cv_score_mean'))
    pd.testing.assert_series_equal(evaluation_result.get('cv_scores'), pd.Series([np.nan] * 3))
    for i in range(automl.data_splitter.get_n_splits()):
        assert np.isnan(evaluation_result['cv_data'][i]['all_objective_scores']['Log Loss Binary'])
    assert 'yeet' in caplog.text


@patch('evalml.objectives.BinaryClassificationObjective.optimize_threshold')
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_train_pipeline_trains_and_tunes_threshold(mock_pipeline_fit, mock_predict_proba, mock_optimize, X_y_binary,
                                                   dummy_binary_pipeline_class):
    X, y = X_y_binary
    _ = EngineBase.train_pipeline(dummy_binary_pipeline_class({}), X, y,
                                  optimize_thresholds=True, objective=LogLossBinary())
    mock_pipeline_fit.assert_called_once()
    mock_optimize.assert_not_called()

    mock_pipeline_fit.reset_mock()
    mock_optimize.reset_mock()

    _ = EngineBase.train_pipeline(dummy_binary_pipeline_class({}), X, y,
                                  optimize_thresholds=True, objective=F1())
    mock_pipeline_fit.assert_called_once()
    mock_optimize.assert_called_once()
