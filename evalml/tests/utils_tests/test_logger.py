from evalml.utils import Logger


def test_logger_verbose():
    logger = Logger()
    assert logger.verbose
    logger = Logger(False)
    assert not logger.verbose
    logger.verbose = True
    assert logger.verbose

def test_logger_log(capsys):
    logger = Logger()
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

