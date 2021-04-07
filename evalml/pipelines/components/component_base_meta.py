

from functools import wraps

from evalml.exceptions import ComponentNotYetFittedError
from evalml.utils.base_meta import BaseMeta


class ComponentBaseMeta(BaseMeta):
    """Metaclass that overrides creating a new component by wrapping methods with validators and setters"""

    @classmethod
    def check_for_fit(cls, method):
        """`check_for_fit` wraps a method that validates if `self._is_fitted` is `True`.
            It raises an exception if `False` and calls and returns the wrapped method if `True`.
        """
        @wraps(method)
        def _check_for_fit(self, X=None, y=None, **kwargs):
            klass = type(self).__name__
            if not self._is_fitted and self.needs_fitting:
                raise ComponentNotYetFittedError(f'This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}.')
            elif method.__name__ == 'inverse_transform':
                return method(self, X, y, **kwargs)
            elif X is None and y is None:
                return method(self, **kwargs)
            elif y is None:
                return method(self, X, **kwargs)
            else:
                return method(self, X, y, **kwargs)
        return _check_for_fit
