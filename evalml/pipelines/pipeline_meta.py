

from functools import wraps

from evalml.exceptions import PipelineNotYetFittedError
from evalml.utils.base_meta import BaseMeta


class PipelineBaseMeta(BaseMeta):
    """Metaclass that overrides creating a new pipeline by wrapping methods with validators and setters"""

    @classmethod
    def check_for_fit(cls, method):
        """`check_for_fit` wraps a method that validates if `self._is_fitted` is `True`.
            It raises an exception if `False` and calls and returns the wrapped method if `True`.
        """
        @wraps(method)
        def _check_for_fit(self, X=None, objective=None):
            klass = type(self).__name__
            if not self._is_fitted:
                raise PipelineNotYetFittedError(f'This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}.')
            if method.__name__ == 'predict_proba':
                return method(self, X)
            elif method.__name__ == 'predict':
                return method(self, X, objective)
            else:
                return method(self)
        return _check_for_fit


class TimeSeriesPipelineBaseMeta(PipelineBaseMeta):
    """Metaclass that overrides creating a new time series pipeline by wrapping methods with validators and setters"""

    @classmethod
    def check_for_fit(cls, method):
        """`check_for_fit` wraps a method that validates if `self._is_fitted` is `True`.
            It raises an exception if `False` and calls and returns the wrapped method if `True`.
        """
        @wraps(method)
        def _check_for_fit(self, X=None, y=None, objective=None):
            klass = type(self).__name__
            if not self._is_fitted:
                raise PipelineNotYetFittedError(f'This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}.')
            if method.__name__ == 'predict_proba':
                return method(self, X, y)
            elif method.__name__ == 'predict':
                return method(self, X, y, objective)
        return _check_for_fit
