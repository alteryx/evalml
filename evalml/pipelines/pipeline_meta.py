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
        def _check_for_fit(self, *args, **kwargs):
            klass = type(self).__name__
            if not self._is_fitted:
                raise PipelineNotYetFittedError(
                    f"This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}."
                )

            return method(self, *args, **kwargs)

        return _check_for_fit
