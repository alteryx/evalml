import importlib


def import_or_raise(library, error_msg=None):
    '''
    Attempts to import the requested library by name.
    If the import fails, raises an ImportError.

    Arguments:
        library (str): the name of the library
        error_msg (str): error message to return if the import fails
    '''
    try:
        return importlib.import_module(library)
    except ImportError:
        if error_msg:
            raise ImportError(error_msg)
        else:
            raise ImportError("Failed to import ".format(library))
