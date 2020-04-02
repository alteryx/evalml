from evalml.utils import *
from colorama import Style


def test_logger_verbose():
    logger = get_logger('evalml')
    logger.error('THIS IS AN ERRORORRORORORORORORORO COPY')

    logger.info('THIS IS A INFO COPY')

    log_title(logger, "HIIIIIIII")
    # logger = Logger()
    # assert logger.verbose
    # logger = Logger(False)
    # assert not logger.verbose
    # logger.verbose = True
    # assert logger.verbose


def test_logger_log(capsys):
    logger = get_logger('evalml')

    # logger = Logger()
    # logger.log('Test message')
    # out, err = capsys.readouterr()
    # assert out == 'Test message\n'
    # assert len(err) == 0

    # logger.log_title('Log title')
    # out, err = capsys.readouterr()
    # assert 'Log title' in out
    # assert len(err) == 0

    # logger.log_subtitle('Log subtitle')
    # out, err = capsys.readouterr()
    # assert 'Log subtitle' in out
    # assert len(err) == 0
