
import logging

from evalml.utils.logger import get_logger, log_subtitle, log_title

logger = get_logger(__name__)


def test_get_logger():
    assert isinstance(logger, logging.Logger)


def test_logger_title(caplog):
    log_title(logger, "Log title")
    assert "Log title" in caplog.text

    caplog.clear()
    log_subtitle(logger, "Log subtitle")
    assert "Log subtitle" in caplog.text


def test_logger_info(caplog):
    logger.info("Test info")
    assert "Test info" in caplog.text
    assert "INFO" in caplog.text


def test_logger_warn(caplog):
    logger.warn("Test warning")
    assert "Test warning" in caplog.text
    assert "WARN" in caplog.text


def test_logger_error(caplog):
    logger.error("Test error")
    assert "Test error" in caplog.text
    assert "ERROR" in caplog.text


def test_logger_critical(caplog):
    logger.critical("Test critical")
    assert "Test critical" in caplog.text
    assert "CRITICAL" in caplog.text
