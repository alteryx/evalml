from evalml.utils import Logger


def test_logger_verbose():
    logger = Logger()
    assert logger.verbose
    logger = Logger(False)
    assert not logger.verbose
    logger.verbose = True
    assert logger.verbose


def test_logger_log(caplog):
    logger = Logger()
    logger.log('Test message')
    out = caplog.text
    assert out == 'Test message\n'

    caplog.clear()
    logger.log_title('Log title')
    out = caplog.text
    assert 'Log title' in out

    caplog.clear()
    logger.log_subtitle('Log subtitle')
    out = caplog.text
    assert 'Log subtitle' in out
