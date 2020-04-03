from colorama import Style

from evalml.utils import *


def test_logger_log_utils(capsys):
    logger = get_logger("evalml", print_std=True)

    log(logger, "Test message")
    out, err = capsys.readouterr()
    assert out == "Test message\n"
    assert len(err) == 0

    log_title(logger, "Log title")
    out, err = capsys.readouterr()
    assert "Log title" in out
    assert len(err) == 0

    log_subtitle(logger, "Log subtitle")
    out, err = capsys.readouterr()
    assert "Log subtitle" in out
    assert len(err) == 0


def test_logger_levels(capsys):
    logger = get_logger("evalml", print_std=True)

    logger.info()
    log(logger, "Test message")
    out, err = capsys.readouterr()
    assert out == "Test message\n"
    assert len(err) == 0

    log_title(logger, "Log title")
    out, err = capsys.readouterr()
    assert "Log title" in out
    assert len(err) == 0

    log_subtitle(logger, "Log subtitle")
    out, err = capsys.readouterr()
    assert "Log subtitle" in out
    assert len(err) == 0
