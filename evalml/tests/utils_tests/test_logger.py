
import logging

from evalml.utils.logger import get_logger, log_subtitle, log_title

logger = get_logger(__name__)


def test_get_logger():
    assert isinstance(logger, logging.Logger)


def test_logger_title(caplog):
    log_title(logger, "Log title")
    out = caplog.text
    assert "Log title" in out

    caplog.clear()
    log_subtitle(logger, "Log subtitle")
    out = caplog.text
    assert "Log subtitle" in out


def test_logger_info(caplog):
    logger.info("Test info")
    assert "Test info" in caplog.messages[0]


def test_logger_warn(caplog):
    logger.warn("Test warning")
    assert "Test warning" in caplog.messages[0]


def test_logger_error(caplog):
    logger.error("Test error")
    assert "Test error" in caplog.messages[0]


def test_logger_critical(caplog):
    logger.critical("Test critical")
    assert "Test critical" in caplog.messages[0]
