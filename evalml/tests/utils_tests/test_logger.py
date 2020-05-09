from evalml.utils import get_logger


def test_logger_log(capsys):
    logger = get_logger(__file__)
    logger.log('Test message')
    out, err = capsys.readouterr()
    assert out == 'Test message\n'
    assert len(err) == 0

    logger.log_title('Log title')
    out, err = capsys.readouterr()
    assert 'Log title' in out
    assert len(err) == 0

    logger.log_subtitle('Log subtitle')
    out, err = capsys.readouterr()
    assert 'Log subtitle' in out
    assert len(err) == 0
