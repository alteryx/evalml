import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from evalml.automl.automl_search import AutoMLSearch
from evalml.automl.engine import evaluate_pipeline, train_pipeline
from evalml.automl.engine.engine_base import JobLogger
from evalml.automl.utils import AutoMLConfig
from evalml.objectives import F1, LogLossBinary
from evalml.preprocessing import split_data


def test_train_and_score_pipelines(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline_class,
    X_y_binary,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_time=1,
        max_batches=1,
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class]
        },
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    pipeline = dummy_binary_pipeline_class({})
    with env.test_context(score_return_value={automl.objective.name: 0.42}):
        evaluation_result = evaluate_pipeline(
            pipeline,
            automl.automl_config,
            automl.X_train,
            automl.y_train,
            logger=MagicMock(),
        ).get("scores")
    assert env.mock_fit.call_count == automl.data_splitter.get_n_splits()
    assert env.mock_score.call_count == automl.data_splitter.get_n_splits()
    assert evaluation_result.get("training_time") is not None
    assert evaluation_result.get("cv_score_mean") == 0.42
    pd.testing.assert_series_equal(
        evaluation_result.get("cv_scores"), pd.Series([0.42] * 3)
    )
    for i in range(automl.data_splitter.get_n_splits()):
        assert (
            evaluation_result["cv_data"][i]["all_objective_scores"]["Log Loss Binary"]
            == 0.42
        )


def test_train_and_score_pipelines_error(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline_class,
    X_y_binary,
    caplog,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_time=1,
        max_batches=1,
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class]
        },
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    pipeline = dummy_binary_pipeline_class({})

    job_log = JobLogger()
    with env.test_context(mock_score_side_effect=Exception("yeet")):
        result = evaluate_pipeline(
            pipeline,
            automl.automl_config,
            automl.X_train,
            automl.y_train,
            logger=job_log,
        )
    evaluation_result, job_log = result.get("scores"), result.get("logger")
    logger = logging.getLogger(__name__)
    job_log.write_to_logger(logger)

    assert env.mock_fit.call_count == automl.data_splitter.get_n_splits()
    assert env.mock_score.call_count == automl.data_splitter.get_n_splits()
    assert evaluation_result.get("training_time") is not None
    assert np.isnan(evaluation_result.get("cv_score_mean"))
    pd.testing.assert_series_equal(
        evaluation_result.get("cv_scores"), pd.Series([np.nan] * 3)
    )
    for i in range(automl.data_splitter.get_n_splits()):
        assert np.isnan(
            evaluation_result["cv_data"][i]["all_objective_scores"]["Log Loss Binary"]
        )
    assert "yeet" in caplog.text


@patch("evalml.automl.engine.engine_base.split_data")
def test_train_pipeline_trains_and_tunes_threshold(
    mock_split_data,
    X_y_binary,
    AutoMLTestEnv,
    dummy_binary_pipeline_class,
):
    X, y = X_y_binary
    mock_split_data.return_value = split_data(
        X, y, "binary", test_size=0.2, random_seed=0
    )
    automl_config = AutoMLConfig(
        None, "binary", LogLossBinary(), [], None, True, None, 0, None, None
    )
    env = AutoMLTestEnv("binary")
    with env.test_context():
        _ = train_pipeline(
            dummy_binary_pipeline_class({}), X, y, automl_config=automl_config
        )

    env.mock_fit.assert_called_once()
    env.mock_optimize_threshold.assert_not_called()
    mock_split_data.assert_not_called()

    automl_config = AutoMLConfig(
        None, "binary", LogLossBinary(), [], F1(), True, None, 0, None, None
    )
    with env.test_context():
        _ = train_pipeline(
            dummy_binary_pipeline_class({}), X, y, automl_config=automl_config
        )
    env.mock_fit.assert_called_once()
    env.mock_optimize_threshold.assert_called_once()
    mock_split_data.assert_called_once()


def test_job_logger_warning_and_error_messages(caplog):
    job_log = JobLogger()
    job_log.warning("This is a warning!")
    job_log.error("This is an error!")
    logger = logging.getLogger(__name__)
    job_log.write_to_logger(logger)

    assert "This is a warning!" in caplog.text
    assert "This is an error!" in caplog.text
