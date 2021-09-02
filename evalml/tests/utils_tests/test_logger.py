import logging
from unittest.mock import patch

import pytest

from evalml import AutoMLSearch
from evalml.utils.logger import (
    get_logger,
    log_subtitle,
    log_title,
    time_elapsed,
)

TEST_LOGGER_NAME = "my_logger"


@pytest.fixture()
def logger_env_cleanup(monkeypatch):
    # need to clear out the logger so logger state isn't shared across the unit tests
    if TEST_LOGGER_NAME in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[TEST_LOGGER_NAME]


def test_get_logger(logger_env_cleanup, capsys, caplog):
    logger = get_logger(TEST_LOGGER_NAME)
    assert isinstance(logger, logging.Logger)

    stdouterr = capsys.readouterr()
    assert "Warning: cannot write debug logs" not in caplog.text
    assert "Exception encountered while setting up debug log file" not in caplog.text
    assert "Warning: cannot write debug logs" not in stdouterr.out
    assert "Exception encountered while setting up debug log file" not in stdouterr.err
    assert "Warning: cannot write debug logs" not in stdouterr.out
    assert "Exception encountered while setting up debug log file" not in stdouterr.err


def test_logger_title(capsys, caplog, logger_env_cleanup):
    logger = get_logger(TEST_LOGGER_NAME)
    log_title(logger, "Log title")
    assert "Log title" in caplog.text

    caplog.clear()
    log_subtitle(logger, "Log subtitle")
    assert "Log subtitle" in caplog.text


def test_logger_info(caplog, logger_env_cleanup):
    logger = get_logger(TEST_LOGGER_NAME)
    logger.info("Test info")
    assert "Test info" in caplog.text
    assert "INFO" in caplog.text


def test_logger_warn(caplog, logger_env_cleanup):
    logger = get_logger(TEST_LOGGER_NAME)
    logger.warning("Test warning")
    assert "Test warning" in caplog.text
    assert "WARN" in caplog.text


def test_logger_error(caplog, logger_env_cleanup):
    logger = get_logger(TEST_LOGGER_NAME)
    logger.error("Test error")
    assert "Test error" in caplog.text
    assert "ERROR" in caplog.text


def test_logger_critical(caplog, logger_env_cleanup):
    logger = get_logger(TEST_LOGGER_NAME)
    logger.critical("Test critical")
    assert "Test critical" in caplog.text
    assert "CRITICAL" in caplog.text


@pytest.mark.parametrize(
    "time_passed,answer",
    [(101199, "28:06:39"), (3660, "1:01:00"), (65, "01:05"), (7, "00:07")],
)
@patch("time.time")
def test_time_elapsed(mock_time, time_passed, answer):
    mock_time.return_value = time_passed
    time = time_elapsed(start_time=0)
    assert time == answer


@pytest.mark.parametrize(
    "type_, allowed_families, number_, number_min_dep",
    [("binary", None, 8, 5), ("multiclass", 2, 2, 2), ("regression", 3, 3, 3)],
)
@pytest.mark.parametrize("verbose", [True, False])
def test_pipeline_count(
    type_,
    allowed_families,
    number_,
    number_min_dep,
    verbose,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    caplog,
    has_minimal_dependencies,
):
    caplog.clear()
    if type_ == "binary":
        X, y = X_y_binary
    elif type_ == "multiclass":
        X, y = X_y_multi
    else:
        X, y = X_y_regression
    if not allowed_families:
        _ = AutoMLSearch(X_train=X, y_train=y, problem_type=type_, verbose=verbose)
    else:
        if allowed_families == 2:
            _ = AutoMLSearch(
                X_train=X,
                y_train=y,
                problem_type=type_,
                allowed_model_families=["random_forest", "decision_tree"],
                verbose=verbose,
            )
        elif allowed_families == 3:
            _ = AutoMLSearch(
                X_train=X,
                y_train=y,
                problem_type=type_,
                allowed_model_families=[
                    "random_forest",
                    "decision_tree",
                    "extra_trees",
                ],
                verbose=verbose,
            )
    if has_minimal_dependencies:
        assert (
            f"{number_min_dep} pipelines ready for search" in caplog.text
        ) == verbose
    else:
        assert (f"{number_} pipelines ready for search" in caplog.text) == verbose
