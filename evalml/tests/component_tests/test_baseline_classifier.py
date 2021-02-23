import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.model_family import ModelFamily
from evalml.pipelines.components import BaselineClassifier
from evalml.utils import get_random_state


def test_baseline_init():
    baseline = BaselineClassifier()
    assert baseline.parameters["strategy"] == "mode"
    assert baseline.model_family == ModelFamily.BASELINE
    assert baseline.classes_ is None


def test_baseline_invalid_strategy():
    with pytest.raises(ValueError):
        BaselineClassifier(strategy="unfortunately invalid strategy")


def test_baseline_y_is_None(X_y_binary):
    X, _ = X_y_binary
    with pytest.raises(ValueError):
        BaselineClassifier().fit(X, y=None)


@pytest.mark.parametrize('data_type', ['pd', 'ww'])
def test_baseline_binary_mode(data_type, make_data_type):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    y = pd.Series([10, 11, 10, 10])
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    clf = BaselineClassifier(strategy="mode")
    fitted = clf.fit(X, y)
    assert isinstance(fitted, BaselineClassifier)
    assert clf.classes_ == [10, 11]
    expected_predictions = pd.Series(np.array([10] * X.shape[0]), dtype="Int64")
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions.to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (X.shape[0], 2)
    expected_predictions_proba = pd.DataFrame({10: [1., 1., 1., 1.], 11: [0., 0., 0., 0.]})
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_binary_random(X_y_binary):
    X, y = X_y_binary
    values = np.unique(y)
    clf = BaselineClassifier(strategy="random", random_seed=0)
    clf.fit(X, y)
    assert clf.classes_ == [0, 1]

    expected_predictions = pd.Series(get_random_state(0).choice(np.unique(y), len(X)), dtype="Int64")
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions.to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    expected_predictions_proba = pd.DataFrame(np.array([[0.5 for i in range(len(values))]] * len(X)))
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_binary_random_weighted(X_y_binary):
    X, y = X_y_binary
    values, counts = np.unique(y, return_counts=True)
    percent_freq = counts.astype(float) / len(y)
    assert percent_freq.sum() == 1.0

    clf = BaselineClassifier(strategy="random_weighted", random_seed=0)
    clf.fit(X, y)

    assert clf.classes_ == [0, 1]
    expected_predictions = pd.Series(get_random_state(0).choice(np.unique(y), len(X), p=percent_freq), dtype="Int64")
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions.to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    expected_predictions_proba = pd.DataFrame(np.array([[percent_freq[i] for i in range(len(values))]] * len(X)))
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_multiclass_mode():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    y = pd.Series([10, 12, 11, 11])
    clf = BaselineClassifier(strategy="mode")
    clf.fit(X, y)

    assert clf.classes_ == [10, 11, 12]
    predictions = clf.predict(X)
    expected_predictions = pd.Series([11] * len(X), dtype="Int64")
    assert_series_equal(expected_predictions, predictions.to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    expected_predictions_proba = pd.DataFrame({10: [0., 0., 0., 0.], 11: [1., 1., 1., 1.], 12: [0., 0., 0., 0.]})
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_multiclass_random(X_y_multi):
    X, y = X_y_multi
    values = np.unique(y)
    clf = BaselineClassifier(strategy="random", random_seed=0)
    clf.fit(X, y)

    assert clf.classes_ == [0, 1, 2]
    expected_predictions = pd.Series(get_random_state(0).choice(np.unique(y), len(X)), dtype="Int64")
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions.to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    assert_frame_equal(pd.DataFrame(np.array([[1. / 3 for i in range(len(values))]] * len(X))), predicted_proba.to_dataframe())
    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_multiclass_random_weighted(X_y_multi):
    X, y = X_y_multi
    values, counts = np.unique(y, return_counts=True)
    percent_freq = counts.astype(float) / len(y)
    assert percent_freq.sum() == 1.0
    clf = BaselineClassifier(strategy="random_weighted", random_seed=0)
    clf.fit(X, y)

    assert clf.classes_ == [0, 1, 2]
    expected_predictions = pd.Series(get_random_state(0).choice(np.unique(y), len(X), p=percent_freq), dtype="Int64")
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions.to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    assert_frame_equal(pd.DataFrame(np.array([[percent_freq[i] for i in range(len(values))]] * len(X))), predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))


def test_baseline_no_mode():
    X = pd.DataFrame([[1, 2, 3, 0, 1]])
    y = pd.Series([1, 0, 2, 0, 1])
    clf = BaselineClassifier()
    clf.fit(X, y)

    assert clf.classes_ == [0, 1, 2]
    expected_predictions = pd.Series([0] * len(X), dtype="Int64")
    predictions = clf.predict(X)
    assert_series_equal(expected_predictions, predictions.to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    assert_frame_equal(pd.DataFrame(np.array([[1.0 if i == 0 else 0.0 for i in range(3)]] * len(X))), predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))
