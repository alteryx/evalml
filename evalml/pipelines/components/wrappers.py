from functools import wraps


def check_for_fit(method):
    @wraps(method)
    def wrapped(self, X):
        if self._has_fit is False:
            raise RuntimeError('Cannot call predict before fit')
        return method(self, X)
    return wrapped


def set_fit(method):
    @wraps(method)
    def wrapped(self, X, y=None):
        self._has_fit = True
        return method(self, X, y)
    return wrapped
