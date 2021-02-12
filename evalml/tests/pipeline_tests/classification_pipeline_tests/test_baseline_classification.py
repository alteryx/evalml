import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.pipelines import BaselineBinaryPipeline, BaselineMulticlassPipeline
from evalml.utils import get_random_state


def test_baseline_binary_random(X_y_binary):
    X, y = X_y_binary
    values = np.unique(y)
    parameters = {
        "Baseline Classifier": {
            "strategy": "random"
        }
    }
    clf = BaselineBinaryPipeline(parameters=parameters)
    clf.fit(X, y)
    expected_predictions = pd.Series(get_random_state(0).choice(np.unique(y), len(X)), dtype="Int64")
    assert_series_equal(expected_predictions, clf.predict(X).to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    expected_predictions_proba = pd.DataFrame(np.array([[0.5 for i in range(len(values))]] * len(X)))
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_binary_random_weighted(X_y_binary):
    X, y = X_y_binary
    values, counts = np.unique(y, return_counts=True)
    percent_freq = counts.astype(float) / len(y)
    assert percent_freq.sum() == 1.0

    parameters = {
        "Baseline Classifier": {
            "strategy": "random_weighted"
        }
    }
    clf = BaselineBinaryPipeline(parameters=parameters)
    clf.fit(X, y)

    expected_predictions = pd.Series(get_random_state(0).choice(np.unique(y), len(X), p=percent_freq), dtype="Int64")
    assert_series_equal(expected_predictions, clf.predict(X).to_series())

    expected_predictions_proba = pd.DataFrame(np.array([[percent_freq[i] for i in range(len(values))]] * len(X)))
    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_binary_mode():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    y = pd.Series([10, 11, 10])
    parameters = {
        "Baseline Classifier": {
            "strategy": "mode"
        }
    }
    clf = BaselineBinaryPipeline(parameters=parameters)
    clf.fit(X, y)

    expected_predictions = pd.Series(np.array([10] * len(X)), dtype="Int64")
    assert_series_equal(expected_predictions, clf.predict(X).to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 2)
    expected_predictions_proba = pd.DataFrame({10: [1., 1., 1., 1.], 11: [0., 0., 0., 0.]})
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_multi_random(X_y_multi):
    X, y = X_y_multi
    values = np.unique(y)
    parameters = {
        "Baseline Classifier": {
            "strategy": "random"
        }
    }
    clf = BaselineMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)

    expected_predictions = pd.Series(get_random_state(0).choice(np.unique(y), len(X)), dtype="Int64")
    assert_series_equal(expected_predictions, clf.predict(X).to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    expected_predictions_proba = pd.DataFrame(np.array([[1. / 3 for i in range(len(values))]] * len(X)))
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())
    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_multi_random_weighted(X_y_multi):
    X, y = X_y_multi
    values, counts = np.unique(y, return_counts=True)
    percent_freq = counts.astype(float) / len(y)
    assert percent_freq.sum() == 1.0

    parameters = {
        "Baseline Classifier": {
            "strategy": "random_weighted"
        }
    }
    clf = BaselineMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)

    expected_predictions = pd.Series(get_random_state(0).choice(np.unique(y), len(X), p=percent_freq), dtype="Int64")
    assert_series_equal(expected_predictions, clf.predict(X).to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    expected_predictions_proba = pd.DataFrame(np.array([[percent_freq[i] for i in range(len(values))]] * len(X)))
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))


def test_baseline_multi_mode():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    y = pd.Series([10, 11, 12, 11])
    parameters = {
        "Baseline Classifier": {
            "strategy": "mode"
        }
    }
    clf = BaselineMulticlassPipeline(parameters=parameters)
    clf.fit(X, y)
    expected_predictions = pd.Series(np.array([11] * len(X)), dtype="Int64")
    assert_series_equal(expected_predictions, clf.predict(X).to_series())

    predicted_proba = clf.predict_proba(X)
    assert predicted_proba.shape == (len(X), 3)
    expected_predictions_proba = pd.DataFrame({10: [0., 0., 0., 0.], 11: [1., 1., 1., 1.], 12: [0., 0., 0., 0.]})
    assert_frame_equal(expected_predictions_proba, predicted_proba.to_dataframe())

    np.testing.assert_allclose(clf.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1]))
