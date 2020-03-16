from evalml.utils import Logger


def test_logger_verbose():
    logger = Logger()
    assert logger.verbose
    logger = Logger(False)
    assert not logger.verbose
    logger.verbose = True
    assert logger.verbose

def test_logger_log():
    logger = Logger()
    logger.log('Test message')
    logger.log_title('Log title')
    logger.log_subtitle('Log subtitle')
