from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evalml.automl import AutoMLSearch
from evalml.automl.engine import SequentialEngine
from evalml.preprocessing import split_data


def test_evaluate_no_data():
    engine = SequentialEngine()
    expected_error = "Dataset has not been loaded into the engine."
    with pytest.raises(ValueError, match=expected_error):
        engine.evaluate_batch([])


def test_add_ensemble_data():
    X = pd.DataFrame({"a": [i for i in range(100)]})
    y = pd.Series([i % 2 for i in range(100)])
    engine = SequentialEngine(X_train=X,
                              y_train=y,
                              automl=None)
    pd.testing.assert_frame_equal(engine.X_train, X)
    assert engine.X_ensemble is None

    X_train, X_ensemble, y_train, y_ensemble = split_data(X, y, problem_type='binary', test_size=0.25)
    engine.add_ensemble(X_train, y_train, X_ensemble, y_ensemble)
    pd.testing.assert_frame_equal(engine.X_train.to_dataframe(), X_train.to_dataframe())
    pd.testing.assert_frame_equal(engine.X_ensemble.to_dataframe(), X_ensemble.to_dataframe())
    pd.testing.assert_series_equal(engine.y_ensemble.to_series(), y_ensemble.to_series())


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_evaluate_batch(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.side_effect = [{'Log Loss Binary': 0.42}] * 3 + [{'Log Loss Binary': 0.5}] * 3
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_batches=1,
                          allowed_pipelines=[dummy_binary_pipeline_class])
    pipelines = [dummy_binary_pipeline_class({'Mock Classifier': {'a': 1}}),
                 dummy_binary_pipeline_class({'Mock Classifier': {'a': 4.2}})]

    mock_should_continue_callback = MagicMock(return_value=True)
    mock_pre_evaluation_callback = MagicMock()
    mock_post_evaluation_callback = MagicMock(side_effect=[123, 456])

    engine = SequentialEngine(X_train=automl.X_train,
                              y_train=automl.y_train,
                              automl=automl,
                              should_continue_callback=mock_should_continue_callback,
                              pre_evaluation_callback=mock_pre_evaluation_callback,
                              post_evaluation_callback=mock_post_evaluation_callback)
    new_pipeline_ids = engine.evaluate_batch(pipelines)

    assert len(pipelines) == 2  # input arg should not have been modified
    assert mock_should_continue_callback.call_count == 3
    assert mock_pre_evaluation_callback.call_count == 2
    assert mock_post_evaluation_callback.call_count == 2
    assert new_pipeline_ids == [123, 456]
    assert mock_pre_evaluation_callback.call_args_list[0][0][0] == pipelines[0]
    assert mock_pre_evaluation_callback.call_args_list[1][0][0] == pipelines[1]
    assert mock_post_evaluation_callback.call_args_list[0][0][0] == pipelines[0]
    assert mock_post_evaluation_callback.call_args_list[0][0][1]['cv_score_mean'] == 0.42
    assert mock_post_evaluation_callback.call_args_list[1][0][0] == pipelines[1]
    assert mock_post_evaluation_callback.call_args_list[1][0][1]['cv_score_mean'] == 0.5


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_evaluate_batch_should_continue(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.side_effect = [{'Log Loss Binary': 0.42}] * 3 + [{'Log Loss Binary': 0.5}] * 3
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_batches=1,
                          allowed_pipelines=[dummy_binary_pipeline_class])
    pipelines = [dummy_binary_pipeline_class({'Mock Classifier': {'a': 1}}),
                 dummy_binary_pipeline_class({'Mock Classifier': {'a': 4.2}})]

    # signal stop after 1st pipeline
    mock_should_continue_callback = MagicMock(side_effect=[True, False])
    mock_pre_evaluation_callback = MagicMock()
    mock_post_evaluation_callback = MagicMock(side_effect=[123, 456])

    engine = SequentialEngine(X_train=automl.X_train,
                              y_train=automl.y_train,
                              automl=automl,
                              should_continue_callback=mock_should_continue_callback,
                              pre_evaluation_callback=mock_pre_evaluation_callback,
                              post_evaluation_callback=mock_post_evaluation_callback)
    new_pipeline_ids = engine.evaluate_batch(pipelines)

    assert len(pipelines) == 2  # input arg should not have been modified
    assert mock_should_continue_callback.call_count == 2
    assert mock_pre_evaluation_callback.call_count == 1
    assert mock_post_evaluation_callback.call_count == 1
    assert new_pipeline_ids == [123]
    assert mock_pre_evaluation_callback.call_args_list[0][0][0] == pipelines[0]
    assert mock_post_evaluation_callback.call_args_list[0][0][0] == pipelines[0]
    assert mock_post_evaluation_callback.call_args_list[0][0][1]['cv_score_mean'] == 0.42

    # no pipelines
    mock_should_continue_callback = MagicMock(return_value=False)
    mock_pre_evaluation_callback = MagicMock()
    mock_post_evaluation_callback = MagicMock(side_effect=[123, 456])

    engine = SequentialEngine(X_train=automl.X_train,
                              y_train=automl.y_train,
                              automl=automl,
                              should_continue_callback=mock_should_continue_callback,
                              pre_evaluation_callback=mock_pre_evaluation_callback,
                              post_evaluation_callback=mock_post_evaluation_callback)
    new_pipeline_ids = engine.evaluate_batch(pipelines)

    assert len(pipelines) == 2  # input arg should not have been modified
    assert mock_should_continue_callback.call_count == 1
    assert mock_pre_evaluation_callback.call_count == 0
    assert mock_post_evaluation_callback.call_count == 0
    assert new_pipeline_ids == []
