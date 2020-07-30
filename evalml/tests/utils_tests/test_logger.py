import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest

from evalml.utils.logger import get_logger, log_subtitle, log_title

TEST_LOGGER_NAME = 'my_logger'


@pytest.fixture()
def logger_env_cleanup(monkeypatch, autouse=True):
    # need to clear out the logger so logger state isn't shared across the unit tests
    if TEST_LOGGER_NAME in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[TEST_LOGGER_NAME]
    # clean up any patches to the logger env var
    assert os.environ.get('EVALML_LOG_FILE') is None
    yield
    monkeypatch.delenv('EVALML_LOG_FILE', raising=False)


def test_get_logger():
    logger = get_logger(TEST_LOGGER_NAME)
    assert isinstance(logger, logging.Logger)


def test_logger_title(caplog):
    logger = get_logger(TEST_LOGGER_NAME)
    log_title(logger, "Log title")
    assert "Log title" in caplog.text

    caplog.clear()
    log_subtitle(logger, "Log subtitle")
    assert "Log subtitle" in caplog.text


def test_logger_info(caplog):
    logger = get_logger(TEST_LOGGER_NAME)
    logger.info("Test info")
    assert "Test info" in caplog.text
    assert "INFO" in caplog.text


def test_logger_warn(caplog):
    logger = get_logger(TEST_LOGGER_NAME)
    logger.warn("Test warning")
    assert "Test warning" in caplog.text
    assert "WARN" in caplog.text


def test_logger_error(caplog):
    logger = get_logger(TEST_LOGGER_NAME)
    logger.error("Test error")
    assert "Test error" in caplog.text
    assert "ERROR" in caplog.text


def test_logger_critical(caplog):
    logger = get_logger(TEST_LOGGER_NAME)
    logger.critical("Test critical")
    assert "Test critical" in caplog.text
    assert "CRITICAL" in caplog.text


@patch('evalml.utils.logger.RotatingFileHandler')
def test_get_logger_default(mock_RotatingFileHandler):
    assert os.environ.get('EVALML_LOG_FILE') is None
    logger = get_logger(TEST_LOGGER_NAME)
    assert len(logger.handlers) == 2
    mock_RotatingFileHandler.assert_called_with(filename=Path("evalml_debug.log"))
    assert len(mock_RotatingFileHandler.mock_calls) == 4
    assert mock_RotatingFileHandler.mock_calls[1] == call().setLevel(logging.DEBUG)


@patch('evalml.utils.logger.RotatingFileHandler')
def test_get_logger_path_valid(mock_RotatingFileHandler, monkeypatch):
    assert os.environ.get('EVALML_LOG_FILE') is None

    with tempfile.TemporaryDirectory() as temp_dir:
        log_file_path = str(Path(temp_dir, 'evalml_debug_custom.log'))
        monkeypatch.setenv('EVALML_LOG_FILE', log_file_path)
        assert os.environ.get('EVALML_LOG_FILE') == log_file_path

        logger = get_logger(TEST_LOGGER_NAME)
        assert len(logger.handlers) == 2
        mock_RotatingFileHandler.assert_called_with(filename=Path(log_file_path))
        assert len(mock_RotatingFileHandler.mock_calls) == 4
        assert mock_RotatingFileHandler.mock_calls[1] == call().setLevel(logging.DEBUG)


@patch('evalml.utils.logger.RotatingFileHandler')
def test_get_logger_path_invalid(mock_RotatingFileHandler, monkeypatch):
    assert os.environ.get('EVALML_LOG_FILE') is None

    with tempfile.TemporaryDirectory() as temp_dir:
        log_file_path = str(Path(temp_dir, 'INVALID', 'PATH', 'DOES_NOT_EXIST', 'evalml_debug_custom.log'))
        monkeypatch.setenv('EVALML_LOG_FILE', log_file_path)
        assert os.environ.get('EVALML_LOG_FILE') == log_file_path

        logger = get_logger(TEST_LOGGER_NAME)
        assert len(logger.handlers) == 1
        assert len(mock_RotatingFileHandler.mock_calls) == 0
        assert mock_RotatingFileHandler.call_count == 0
