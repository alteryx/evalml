
import logging

from evalml.utils import Logger


def test_get_logger():
    logger = Logger()
    assert isinstance(logger.get_logger(), logging.Logger)


def test_logger_levels():
    logger = Logger("DEBUG")
    assert logger.level == "DEBUG"
    assert logger.get_logger().getEffectiveLevel() == logging.DEBUG

    logger = Logger()
    assert logger.level == "INFO"
    assert logger.get_logger().getEffectiveLevel() == logging.INFO

    logger = Logger("WARNING")
    assert logger.level == "WARNING"
    assert logger.get_logger().getEffectiveLevel() == logging.WARN

    logger = Logger("ERROR")
    assert logger.level == "ERROR"
    assert logger.get_logger().getEffectiveLevel() == logging.ERROR

    logger = Logger("CRITICAL")
    assert logger.level == "CRITICAL"
    assert logger.get_logger().getEffectiveLevel() == logging.CRITICAL


def test_logger_log(caplog, capsys):
    logger = Logger()
    logger.log("Test message")
    assert caplog.messages[0] == "Test message\n"

    caplog.clear()
    logger.log("Test message", new_line=False)
    assert caplog.messages[0] == "Test message"

    caplog.clear()
    logger.log("Test message", new_line=False, print_stdout=True)
    assert caplog.messages[0] == "Test message"
    out, err = capsys.readouterr()
    assert out == "Test message"
    assert err == ""


def test_logger_title(caplog):
    logger = Logger()
    logger.log_title("Log title")
    out = caplog.text
    assert "Log title" in out

    caplog.clear()
    logger.log_subtitle("Log subtitle")
    out = caplog.text
    assert "Log subtitle" in out


def test_logger_print(caplog, capsys):
    logger = Logger()
    logger.print("Test print without newline", new_line=False, log=True)
    out, err = capsys.readouterr()
    assert "Test print without newline" in caplog.messages[0]
    assert out == "Test print without newline"
    assert len(err) == 0

    caplog.clear()
    logger.print("Test print with newline", new_line=True, log=True)
    out, err = capsys.readouterr()
    assert "Test print with newline" in caplog.messages[0]
    assert out == "Test print with newline\n"
    assert len(err) == 0

    caplog.clear()
    logger.print("Test print", log=False)
    assert "Test print" not in caplog.text


def test_logger_warn(caplog):
    logger = Logger()
    logger.warn("Test warning", stack_info=True)
    assert "Test warning" in caplog.messages[0]
    assert "Stack (most recent call last):" in caplog.text

    caplog.clear()
    logger.warn("Test warning", stack_info=False)
    assert "Test warning" in caplog.messages[0]
    assert "Stack (most recent call last):" not in caplog.text


def test_logger_error(caplog):
    logger = Logger()
    logger.error("Test error", stack_info=True)
    assert "Test error" in caplog.messages[0]
    assert "Stack (most recent call last):" in caplog.text

    caplog.clear()
    logger.warn("Test error", stack_info=False)
    assert "Test error" in caplog.messages[0]
    assert "Stack (most recent call last):" not in caplog.text


def test_logger_critical(caplog):
    logger = Logger()
    logger.critical("Test critical", stack_info=True)
    assert "Test critical" in caplog.messages[0]
    assert "Stack (most recent call last):" in caplog.text

    caplog.clear()
    logger.warn("Test critical", stack_info=False)
    assert "Test critical" in caplog.messages[0]
    assert "Stack (most recent call last):" not in caplog.text
