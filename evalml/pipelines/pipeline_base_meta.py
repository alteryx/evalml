

import inspect
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
        def _check_for_fit(self, X=None, y=None, objective=None):
            klass = type(self).__name__
            if not self._is_fitted:
                raise PipelineNotYetFittedError(f'This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}.')
            elif X is None and y is None:
                return method(self)
            elif y is None:
                return method(self, X)
            # For time series classification pipelines, predict will take X, y, objective
            elif len(inspect.getfullargspec(method).args) == 4:
                return method(self, X, y, objective)
            # For other pipelines, predict will take X, y or X, objective
            else:
                return method(self, X, y)
        return _check_for_fit
