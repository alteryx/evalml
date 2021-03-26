from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.automl import AutoMLSearch
from evalml.automl.engine import SequentialEngine
from evalml.exceptions import PipelineScoreError
from evalml.objectives import F1
from evalml.pipelines.components.ensemble import StackedEnsembleClassifier
from evalml.pipelines.utils import make_pipeline_from_components
from evalml.preprocessing import split_data
from evalml.utils.woodwork_utils import infer_feature_types


def test_evaluate_no_data():
    engine = SequentialEngine()
    expected_error = "Dataset has not been loaded into the engine."
    with pytest.raises(ValueError, match=expected_error):
        engine.evaluate_batch([])

    with pytest.raises(ValueError, match=expected_error):
        engine.train_batch([])


def test_add_ensemble_data():
    X = pd.DataFrame({"a": [i for i in range(100)]})
    y = pd.Series([i % 2 for i in range(100)])
    engine = SequentialEngine(X_train=X,
                              y_train=y,
                              automl=None)
    pd.testing.assert_frame_equal(engine.X_train, X)
    assert engine.ensembling_indices is None

    training_indices, ensembling_indices, _, _ = split_data(ww.DataTable(np.arange(X.shape[0])), y, problem_type='binary', test_size=0.2, random_seed=0)
    training_indices, ensembling_indices = training_indices.to_dataframe()[0].tolist(), ensembling_indices.to_dataframe()[0].tolist()
    engine = SequentialEngine(X_train=X,
                              y_train=y,
                              ensembling_indices=ensembling_indices,
                              automl=None)
    pd.testing.assert_frame_equal(engine.X_train, X)
    assert engine.ensembling_indices == ensembling_indices


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_ensemble_data(mock_fit, mock_score, dummy_binary_pipeline_class, stackable_classifiers):
    X = pd.DataFrame({"a": [i for i in range(100)]})
    y = pd.Series([i % 2 for i in range(100)])
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=19, ensembling=True, _ensembling_split_size=0.25)
    mock_should_continue_callback = MagicMock(return_value=True)
    mock_pre_evaluation_callback = MagicMock()
    mock_post_evaluation_callback = MagicMock()

    training_indices, ensembling_indices, _, _ = split_data(ww.DataTable(np.arange(X.shape[0])), y, problem_type='binary', test_size=0.25, random_seed=0)
    training_indices, ensembling_indices = training_indices.to_dataframe()[0].tolist(), ensembling_indices.to_dataframe()[0].tolist()

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
    engine = SequentialEngine(X_train=automl.X_train,
                              y_train=automl.y_train,
                              automl=automl)

    def make_pipeline_name(index):
        class DummyPipeline(dummy_binary_pipeline_class):
            custom_name = f"Pipeline {index}"
        return DummyPipeline({'Mock Classifier': {'a': index}})

    pipelines = [make_pipeline_name(i) for i in range(len(pipeline_fit_side_effect) - 1)]
    ensemble_input_pipelines = [make_pipeline_from_components([classifier], problem_type="binary") for classifier in stackable_classifiers[:2]]
    ensemble = make_pipeline_from_components([StackedEnsembleClassifier(ensemble_input_pipelines, n_jobs=1)], problem_type="binary")
    pipelines.append(ensemble)

    def train_batch_and_check():
        caplog.clear()
        with patch('evalml.pipelines.BinaryClassificationPipeline.fit') as mock_fit:
            mock_fit.side_effect = pipeline_fit_side_effect
            trained_pipelines = engine.train_batch(pipelines)
            assert len(trained_pipelines) == len(pipeline_fit_side_effect) - len(exceptions_to_check)
            assert mock_fit.call_count == len(pipeline_fit_side_effect)
            for exception in exceptions_to_check:
                assert exception in caplog.text

    # Test training before search is run
    train_batch_and_check()

    # Test training after search.
    automl.search()

    train_batch_and_check()


@patch('evalml.automl.EngineBase.train_pipeline')
def test_train_batch_performs_undersampling(mock_train, X_y_binary, dummy_binary_pipeline_class):
    X, y = X_y_binary
    X = ww.DataTable(X)
    y = ww.DataColumn(y)

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_iterations=2,
                          train_best_pipeline=False, n_jobs=1)
    engine = SequentialEngine(X_train=automl.X_train,
                              y_train=automl.y_train,
                              automl=automl)

    train_indices = automl.data_splitter.transform_sample(X, y)
    X_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]

    pipelines = [dummy_binary_pipeline_class({})]
    engine.train_batch(pipelines)

    args, kwargs = mock_train.call_args  # args are (pipeline, X, y, optimize_thresholds, objective)
    pd.testing.assert_frame_equal(X_train.to_dataframe(), args[1].to_dataframe())
    pd.testing.assert_series_equal(y_train.to_series(), args[2].to_series())


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
                          allowed_pipelines=[dummy_binary_pipeline_class])

    engine = SequentialEngine(X_train=automl.X_train, y_train=automl.y_train, automl=automl)

    def make_pipeline_name(index):
        class DummyPipeline(dummy_binary_pipeline_class):
            custom_name = f"Pipeline {index}"
        return DummyPipeline({'Mock Classifier': {'a': index}})

    pipelines = [make_pipeline_name(i) for i in range(len(pipeline_score_side_effect) - 1)]
    ensemble_input_pipelines = [make_pipeline_from_components([classifier], problem_type="binary") for classifier in stackable_classifiers[:2]]
    ensemble = make_pipeline_from_components([StackedEnsembleClassifier(ensemble_input_pipelines, n_jobs=1)],
                                             problem_type="binary")
    pipelines.append(ensemble)

    def score_batch_and_check():
        caplog.clear()
        with patch('evalml.pipelines.BinaryClassificationPipeline.score') as mock_score:
            mock_score.side_effect = pipeline_score_side_effect

            scores = engine.score_batch(pipelines, X, y, objectives=["Log Loss Binary", "F1", "AUC"])
            assert scores == expected_scores
            for exception in exceptions_to_check:
                assert exception in caplog.text

    # Test scoring before search
    score_batch_and_check()

    automl.search()

    # Test scoring after search
    score_batch_and_check()


def test_score_batch_before_fitting_yields_error_nan_scores(X_y_binary, dummy_binary_pipeline_class, caplog):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_iterations=1,
                          allowed_pipelines=[dummy_binary_pipeline_class])

    engine = SequentialEngine(X_train=automl.X_train, y_train=automl.y_train, automl=automl)

    scored_pipelines = engine.score_batch([dummy_binary_pipeline_class({})], X, y, objectives=["Log Loss Binary",
                                                                                               F1()])
    scored_pipelines["Mock Binary Classification Pipeline"] = {"Log Loss Binary": np.nan,
                                                               "F1": np.nan}

    assert "Score error for Mock Binary Classification Pipeline" in caplog.text
    assert "This LabelEncoder instance is not fitted yet." in caplog.text
