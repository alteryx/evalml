

from abc import ABCMeta
from functools import wraps


class BaseMeta(ABCMeta):
    """Metaclass that overrides creating a new component or pipeline by wrapping methods with validators and setters"""

    @classmethod
    def set_fit(cls, method):
        @wraps(method)
        def _set_fit(self, X, y=None):
            return_value = method(self, X, y)
            self._is_fitted = True
            return return_value
        return _set_fit

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
