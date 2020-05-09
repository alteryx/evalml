from evalml.utils import get_logger, log_subtitle, log_title


def test_logger_log(capsys):
    logger = get_logger(__file__)
    logger.info('Test message')
    out, err = capsys.readouterr()
    assert out == 'Test message\n'
    assert len(err) == 0

    log_title(logger, 'Log title')
    out, err = capsys.readouterr()
    assert 'Log title' in out
    assert len(err) == 0

    log_subtitle(logger, 'Log subtitle')
    out, err = capsys.readouterr()
    assert 'Log subtitle' in out
    assert len(err) == 0
