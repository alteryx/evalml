from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evalml.automl import AutoMLSearch
from evalml.automl.engine import SequentialEngine
from evalml.pipelines.components.ensemble import StackedEnsembleClassifier
from evalml.pipelines.utils import make_pipeline_from_components
from evalml.preprocessing import TrainingValidationSplit
from evalml.utils.woodwork_utils import infer_feature_types


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
    assert engine.ensembling_indices is None

    for training_indices, ensembling_indices in TrainingValidationSplit(test_size=0.2, shuffle=True, stratify=y, random_state=0).split(X, y):
        engine = SequentialEngine(X_train=X,
                                  y_train=y,
                                  ensembling_indices=ensembling_indices,
                                  automl=None)
        pd.testing.assert_frame_equal(engine.X_train, X)
        assert all(engine.ensembling_indices == ensembling_indices)


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_ensemble_data(mock_fit, mock_score, dummy_binary_pipeline_class, stackable_classifiers):
    X = pd.DataFrame({"a": [i for i in range(100)]})
    y = pd.Series([i % 2 for i in range(100)])
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=19, ensembling=True, _ensembling_split_size=0.25)
    mock_should_continue_callback = MagicMock(return_value=True)
    mock_pre_evaluation_callback = MagicMock()
    mock_post_evaluation_callback = MagicMock()
    for training_indices, ensembling_indices in TrainingValidationSplit(test_size=0.25, shuffle=True, stratify=y, random_state=0).split(X, y):
        engine = SequentialEngine(X_train=infer_feature_types(X),
                                  y_train=infer_feature_types(y),
                                  ensembling_indices=ensembling_indices,
                                  automl=automl,
                                  should_continue_callback=mock_should_continue_callback,
                                  pre_evaluation_callback=mock_pre_evaluation_callback,
                                  post_evaluation_callback=mock_post_evaluation_callback)
        pipeline1 = [dummy_binary_pipeline_class({'Mock Classifier': {'a': 1}})]
        engine.evaluate_batch(pipeline1)
        # check the fit length is correct, taking into account the data splits
        assert len(mock_fit.call_args[0][0]) == int(2 / 3 * len(training_indices))

        input_pipelines = [make_pipeline_from_components([classifier], problem_type='binary')
                           for classifier in stackable_classifiers]
        pipeline2 = [make_pipeline_from_components([StackedEnsembleClassifier(input_pipelines, n_jobs=1)], problem_type='binary', custom_name="Stacked Ensemble Classification Pipeline")]
        engine.evaluate_batch(pipeline2)
        assert len(mock_fit.call_args[0][0]) == int(2 / 3 * len(ensembling_indices))


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
