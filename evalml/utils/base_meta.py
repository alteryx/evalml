"""Metaclass that overrides creating a new component or pipeline by wrapping methods with validators and setters."""
from abc import ABCMeta
from functools import wraps


class BaseMeta(ABCMeta):
    """Metaclass that overrides creating a new component or pipeline by wrapping methods with validators and setters."""

    FIT_METHODS = ["fit", "fit_transform"]
    METHODS_TO_CHECK = ["predict", "predict_proba", "transform", "inverse_transform"]
    PROPERTIES_TO_CHECK = ["feature_importance"]

    @classmethod
    def set_fit(cls, method):
        """Wrapper for the fit method."""

        @wraps(method)
        def _set_fit(self, X, y=None):
            return_value = method(self, X, y)
            self._is_fitted = True
            return return_value

        return _set_fit

    def __new__(cls, name, bases, dct):
        """Create a new instance."""
        for attribute in dct:
            if attribute in cls.FIT_METHODS:
                dct[attribute] = cls.set_fit(dct[attribute])
            if attribute in cls.METHODS_TO_CHECK:
                dct[attribute] = cls.check_for_fit(dct[attribute])
            if attribute in cls.PROPERTIES_TO_CHECK:
                property_orig = dct[attribute]
                dct[attribute] = property(
                    cls.check_for_fit(property_orig.__get__),
                    property_orig.__set__,
                    property_orig.__delattr__,
                    property_orig.__doc__,
                )
        return super().__new__(cls, name, bases, dct)
