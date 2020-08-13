

from abc import ABCMeta
from functools import wraps

from evalml.exceptions import (
    ComponentNotYetFittedError,
    PipelineNotYetFittedError
)


class BaseMeta(ABCMeta):
    """Metaclass that overrides creating a new component by wrapping method with validators and setters"""
    from evalml.exceptions import ComponentNotYetFittedError

    NO_FITTING_REQUIRED = ['DropColumns', 'SelectColumns']
    error_to_throw = None

    @classmethod
    def set_fit(cls, method):
        @wraps(method)
        def _set_fit(self, X, y=None):
            return_value = method(self, X, y)
            self._is_fitted = True
            return return_value
        return _set_fit

    @classmethod
    def check_for_fit(cls, method):
        """`check_for_fit` wraps a method that validates if `self._is_fitted` is `True`.
            It raises an exception if `False` and calls and returns the wrapped method if `True`.
        """
        @wraps(method)
        def _check_for_fit(self, X=None, y=None):
            klass = type(self).__name__
            if not self._is_fitted and klass not in cls.NO_FITTING_REQUIRED:
                raise cls.error_to_throw(f'This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}.')
            elif X is None and y is None:
                return method(self)
            elif y is None:
                return method(self, X)
            else:
                return method(self, X, y)
        return _check_for_fit

    def __new__(cls, name, bases, dct):
        if 'predict' in dct:
            dct['predict'] = cls.check_for_fit(dct['predict'])
        if 'predict_proba' in dct:
            dct['predict_proba'] = cls.check_for_fit(dct['predict_proba'])
        if 'transform' in dct:
            dct['transform'] = cls.check_for_fit(dct['transform'])
        if 'feature_importance' in dct:
            fi = dct['feature_importance']
            new_fi = property(cls.check_for_fit(fi.__get__), fi.__set__, fi.__delattr__)
            dct['feature_importance'] = new_fi
        if 'fit' in dct:
            dct['fit'] = cls.set_fit(dct['fit'])
        if 'fit_transform' in dct:
            dct['fit_transform'] = cls.set_fit(dct['fit_transform'])
        return super().__new__(cls, name, bases, dct)


class ComponentBaseMeta(BaseMeta):
    error_to_throw = ComponentNotYetFittedError


class PipelineBaseMeta(BaseMeta):
    error_to_throw = PipelineNotYetFittedError
