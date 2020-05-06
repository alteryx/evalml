import pytest

from evalml.pipelines.components import ZeroRClassifier


def test_access_without_fit(X_y):
    X, _ = X_y
    clf = ZeroRClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError):
        clf.feature_importances


def test_y_is_None(X_y):
    X, _ = X_y
    with pytest.raises(ValueError):
        ZeroRClassifier().fit(X, y=None)


def test_multiclass(X_y_multi):
    pass


def test_no_mode():
    pass
