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
    dummy_binary_pipeline,
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
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 0.42}):
        evaluation_result = evaluate_pipeline(
            dummy_binary_pipeline,
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
        evaluation_result.get("cv_scores"),
        pd.Series([0.42] * 3),
    )
    for i in range(automl.data_splitter.get_n_splits()):
        assert (
            evaluation_result["cv_data"][i]["all_objective_scores"]["Log Loss Binary"]
            == 0.42
        )


def test_train_and_score_pipelines_error(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline,
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
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")

    job_log = JobLogger()
    with env.test_context(mock_score_side_effect=Exception("yeet")):
        result = evaluate_pipeline(
            dummy_binary_pipeline,
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
        evaluation_result.get("cv_scores"),
        pd.Series([np.nan] * 3),
    )
    for i in range(automl.data_splitter.get_n_splits()):
        assert np.isnan(
            evaluation_result["cv_data"][i]["all_objective_scores"]["Log Loss Binary"],
        )
    assert "yeet" in caplog.text


@patch("evalml.automl.engine.engine_base.split_data")
def test_train_pipeline_trains_and_tunes_threshold(
    mock_split_data,
    X_y_binary,
    AutoMLTestEnv,
    dummy_binary_pipeline,
):
    X, y = X_y_binary
    mock_split_data.return_value = split_data(
        X,
        y,
        "binary",
        test_size=0.2,
        random_seed=0,
    )
    automl_config = AutoMLConfig(
        None,
        "binary",
        LogLossBinary(),
        [],
        None,
        True,
        None,
        0,
        None,
        None,
        {},
    )
    env = AutoMLTestEnv("binary")
    with env.test_context():
        _ = train_pipeline(dummy_binary_pipeline, X, y, automl_config=automl_config)

    env.mock_fit.assert_called_once()
    env.mock_optimize_threshold.assert_not_called()
    mock_split_data.assert_not_called()

    automl_config = AutoMLConfig(
        None,
        "binary",
        LogLossBinary(),
        [],
        F1(),
        True,
        None,
        0,
        None,
        None,
        {},
    )
    with env.test_context():
        _ = train_pipeline(dummy_binary_pipeline, X, y, automl_config=automl_config)
    env.mock_fit.assert_called_once()
    env.mock_optimize_threshold.assert_called_once()
    mock_split_data.assert_called_once()


def test_train_pipeline_trains_and_tunes_threshold_ts(
    ts_data,
    dummy_ts_binary_tree_classifier_pipeline_class,
):
    X, _, y = ts_data(
        train_features_index_dt=False,
        train_target_index_dt=False,
        no_features=True,
        problem_type="time series binary",
    )

    params = {"gap": 1, "max_delay": 1, "forecast_horizon": 1, "time_index": "date"}
    ts_binary = dummy_ts_binary_tree_classifier_pipeline_class(
        parameters={"pipeline": params},
    )
    assert ts_binary.threshold is None

    automl_config = AutoMLConfig(
        None,
        "time series binary",
        LogLossBinary(),
        [],
        F1(),
        True,
        None,
        0,
        None,
        None,
        {},
    )
    cv_pipeline, _ = train_pipeline(ts_binary, X, y, automl_config=automl_config)
    assert cv_pipeline.threshold is not None


def test_job_logger_warning_and_error_messages(caplog):
    job_log = JobLogger()
    job_log.warning("This is a warning!")
    job_log.error("This is an error!")
    logger = logging.getLogger(__name__)
    job_log.write_to_logger(logger)

    assert "This is a warning!" in caplog.text
    assert "This is an error!" in caplog.text


def test_train_pipelines_cache(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline,
    X_y_binary,
    caplog,
):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    automl_config = AutoMLConfig(
        None,
        "binary",
        LogLossBinary(),
        [],
        None,
        True,
        None,
        0,
        None,
        None,
        {},
    )
    env = AutoMLTestEnv("binary")
    with env.test_context():
        res = train_pipeline(
            dummy_binary_pipeline,
            X,
            y,
            automl_config=automl_config,
            get_hashes=False,
        )
    assert isinstance(res, tuple)
    assert res[1] is None

    with env.test_context():
        res = train_pipeline(
            dummy_binary_pipeline,
            X,
            y,
            automl_config=automl_config,
            get_hashes=True,
        )
    assert isinstance(res, tuple)
    assert res[1] == hash(tuple(X.index))


def test_train_and_score_pipelines_cache(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline,
    X_y_binary,
):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_time=1,
        max_batches=1,
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 0.42}):
        evaluation_result = evaluate_pipeline(
            dummy_binary_pipeline,
            automl.automl_config,
            automl.X_train,
            automl.y_train,
            logger=MagicMock(),
        ).get("cached_data")
    assert evaluation_result
    assert len(evaluation_result) == automl.data_splitter.n_splits
