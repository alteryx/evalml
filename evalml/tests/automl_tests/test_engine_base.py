from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import woodwork as ww

from evalml.automl.automl_search import AutoMLSearch
from evalml.automl.engine import evaluate_pipeline, train_pipeline
from evalml.automl.engine.engine_base import JobLogger
from evalml.objectives import F1, LogLossBinary
from evalml.pipelines import (
    BinaryClassificationPipeline,
    StackedEnsembleClassifier
)
from evalml.preprocessing import split_data
from evalml.utils import get_logger


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_train_and_score_pipelines(mock_fit, mock_score, dummy_binary_pipeline_class, X_y_binary):
    X, y = X_y_binary
    mock_score.return_value = {'Log Loss Binary': 0.42}
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_batches=1,
                          allowed_pipelines=[dummy_binary_pipeline_class({})])
    pipeline = dummy_binary_pipeline_class({})
    evaluation_result = evaluate_pipeline(pipeline, automl, automl.X_train, automl.y_train, logger=MagicMock()).get("scores")
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
                          allowed_pipelines=[dummy_binary_pipeline_class({})])
    pipeline = dummy_binary_pipeline_class({})

    job_log = JobLogger()
    result = evaluate_pipeline(pipeline, automl, automl.X_train, automl.y_train, logger=job_log)
    evaluation_result, job_log = result.get("scores"), result.get("logger")
    logger = get_logger(__file__)
    job_log.write_to_logger(logger)

    assert mock_fit.call_count == automl.data_splitter.get_n_splits()
    assert mock_score.call_count == automl.data_splitter.get_n_splits()
    assert evaluation_result.get('training_time') is not None
    assert np.isnan(evaluation_result.get('cv_score_mean'))
    pd.testing.assert_series_equal(evaluation_result.get('cv_scores'), pd.Series([np.nan] * 3))
    for i in range(automl.data_splitter.get_n_splits()):
        assert np.isnan(evaluation_result['cv_data'][i]['all_objective_scores']['Log Loss Binary'])
    assert 'yeet' in caplog.text


@patch('evalml.objectives.BinaryClassificationObjective.optimize_threshold')
@patch('evalml.pipelines.BinaryClassificationPipeline._encode_targets', side_effect=lambda y: y)
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
@patch('evalml.automl.engine.engine_base.split_data')
def test_train_pipeline_trains_and_tunes_threshold(mock_split_data, mock_pipeline_fit,
                                                   mock_predict_proba, mock_encode_targets, mock_optimize, X_y_binary,
                                                   dummy_binary_pipeline_class):
    X, y = X_y_binary
    mock_split_data.return_value = split_data(X, y, "binary", test_size=0.2, random_seed=0)

    _ = train_pipeline(dummy_binary_pipeline_class({}), X, y,
                       optimize_thresholds=True, objective=LogLossBinary())

    mock_pipeline_fit.assert_called_once()
    mock_optimize.assert_not_called()
    mock_split_data.assert_not_called()

    mock_pipeline_fit.reset_mock()
    mock_optimize.reset_mock()
    mock_split_data.reset_mock()

    _ = train_pipeline(dummy_binary_pipeline_class({}), X, y,
                       optimize_thresholds=True, objective=F1())
    mock_pipeline_fit.assert_called_once()
    mock_optimize.assert_called_once()
    mock_split_data.assert_called_once()


@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_evaluate_pipeline_handles_ensembling_indices(mock_fit, mock_score, dummy_binary_pipeline_class,
                                                      stackable_classifiers):
    X = ww.DataTable(pd.DataFrame({"a": [i for i in range(100)]}))
    y = ww.DataColumn(pd.Series([i % 2 for i in range(100)]))

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_batches=19, ensembling=True, _ensembling_split_size=0.25)

    training_indices, ensembling_indices, _, _ = split_data(ww.DataTable(np.arange(X.shape[0])), y, problem_type='binary', test_size=0.25, random_seed=0)
    training_indices, ensembling_indices = training_indices.to_dataframe()[0].tolist(), ensembling_indices.to_dataframe()[0].tolist()

    pipeline1 = dummy_binary_pipeline_class({'Mock Classifier': {'a': 1}})

    _ = evaluate_pipeline(pipeline1, automl, X, y, logger=MagicMock())
    # check the fit length is correct, taking into account the data splits
    assert len(mock_fit.call_args[0][0]) == int(2 / 3 * len(training_indices))

    input_pipelines = [BinaryClassificationPipeline([classifier]) for classifier in stackable_classifiers]
    ensemble = BinaryClassificationPipeline([StackedEnsembleClassifier],
                                            parameters={"Stacked Ensemble Classifier": {"input_pipelines": input_pipelines, "n_jobs": 1}})

    _ = evaluate_pipeline(ensemble, automl, X, y, logger=MagicMock())
    assert len(mock_fit.call_args[0][0]) == int(2 / 3 * len(ensembling_indices))


def test_job_logger_warning_and_error_messages(caplog):
    job_log = JobLogger()
    job_log.warning("This is a warning!")
    job_log.error("This is an error!")
    logger = get_logger(__file__)
    job_log.write_to_logger(logger)

    assert "This is a warning!" in caplog.text
    assert "This is an error!" in caplog.text
