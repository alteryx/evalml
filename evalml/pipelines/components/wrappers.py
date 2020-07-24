from functools import wraps

from evalml.exceptions import UnfitComponentError

NO_FITTING_REQUIRED = ['DropColumns', 'SelectColumns']


def check_for_fit(method):
    @wraps(method)
    def _check_for_fit(self, X=None, y=None):
        klass = type(self).__name__
        if self._has_fit is False and klass not in NO_FITTING_REQUIRED:
            raise UnfitComponentError('You must fit {} before calling {}.'.format(klass, method.__name__))
        if y is None and X is None:
            return method(self)
        elif y is None:
            return method(self, X)
        else:
            return method(self, X, y)
    return _check_for_fit


def check_for_fit_properties(method):
    @wraps(method)
    def _check_for_fit_properties(self, *args, **kwargs):
        klass = type(self).__name__
        if self._has_fit is False and klass not in NO_FITTING_REQUIRED:
            raise UnfitComponentError('You must fit {} before calling {}.'.format(klass, method.__name__))
        return method(self, *args, **kwargs)
    return _check_for_fit_properties


def set_fit(method):
    @wraps(method)
    def _set_fit(self, X, y=None):
        try:
            return_value = method(self, X, y)
        except Exception as e:
            raise e
        self._has_fit = True
        return return_value
    return _set_fit
