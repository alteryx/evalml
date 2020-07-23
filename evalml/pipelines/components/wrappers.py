from functools import wraps

NO_FITTING_REQUIRED = ['DropColumns', 'SelectColumns']


def check_for_fit(method):
    @wraps(method)
    def _check_for_fit(self, X, y=None):
        klass = type(self).__name__
        if self._has_fit is False and klass not in NO_FITTING_REQUIRED:
            raise RuntimeError('You must fit before calling predict/predict_proba/transform.')
        if y is None:
            return method(self, X)
        else:
            return method(self, X, y)
    return _check_for_fit


def set_fit(method):
    @wraps(method)
    def _set_fit(self, X, y=None):
        self._has_fit = True
        return method(self, X, y)
    return _set_fit
