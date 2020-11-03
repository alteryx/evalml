from evalml.utils.logger import get_logger

logger = get_logger(__file__)


def silent_error_callback(automl, exception):
    """No-op"""
    pass


def log_error_callback(automl, exception):
    """doesn't throw, but logs. default behavior. not sure if defining this here will mess things up"""
    logger.error(f'AutoML search encountered an exception: {str(exception)}')


def raise_error_callback(automl, exception):
    """throws"""
    logger.error(f'AutoML search got fatal exception: {str(exception)}')
    raise exception


def log_and_save_error_callback(automl, exception):
    """doesn't throw, but adds exception to a list in results"""
    logger.warning(f'AutoML search encountered an exception: {str(exception)}')
    automl._results['errors'] = automl._results.get('errors', [])
    automl._results['errors'].append(exception)


def raise_and_save_error(automl, exception):
    """save exception and throw"""
    logger.warning(f'AutoML search encountered an exception: {str(exception)}')
    automl._results['errors'] = [exception]
    raise exception
