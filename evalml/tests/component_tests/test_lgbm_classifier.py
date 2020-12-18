from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines import LightGBMClassifier
from evalml.problem_types import ProblemTypes
from evalml.utils import SEED_BOUNDS

lgbm = importorskip('lightgbm', reason='Skipping test because lightgbm not installed')


def test_model_family():
    assert LightGBMClassifier.model_family == ModelFamily.LIGHTGBM


def test_problem_types():
    assert set(LightGBMClassifier.supported_problem_types) == {ProblemTypes.MULTICLASS, ProblemTypes.BINARY,
                                                               ProblemTypes.TIME_SERIES_MULTICLASS,
                                                               ProblemTypes.TIME_SERIES_BINARY}


def test_lightgbm_classifier_random_state_bounds_seed(X_y_binary):
    """ensure lightgbm's RNG doesn't fail for the min/max bounds we support on user-inputted random seeds"""
    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    clf = LightGBMClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.min_bound, n_jobs=1)
    clf.fit(X, y)
    clf = LightGBMClassifier(n_estimators=1, max_depth=1, random_state=SEED_BOUNDS.max_bound, n_jobs=1)
    clf.fit(X, y)


def test_lightgbm_classifier_random_state_bounds_rng(X_y_binary):
    """when a RNG is inputted for random_state, ensure the sample we take to get a random seed for lightgbm is in lightgbm's supported range"""

    def make_mock_random_state(return_value):

        class MockRandomState(np.random.RandomState):

            def randint(self, min_bound, max_bound):
                return return_value
        return MockRandomState()

    X, y = X_y_binary
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y)
    rng = make_mock_random_state(LightGBMClassifier.SEED_MIN)
    clf = LightGBMClassifier(n_estimators=1, max_depth=1, random_state=rng, n_jobs=1)
    clf.fit(X, y)
    rng = make_mock_random_state(LightGBMClassifier.SEED_MAX)
    clf = LightGBMClassifier(n_estimators=1, max_depth=1, random_state=rng, n_jobs=1)
    clf.fit(X, y)


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    sk_clf = lgbm.sklearn.LGBMClassifier(random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = LightGBMClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    clf = lgbm.sklearn.LGBMClassifier(random_state=0)
    clf.fit(X, y)
    y_pred_sk = clf.predict(X)
    y_pred_proba_sk = clf.predict_proba(X)

    clf = LightGBMClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_feature_importance(X_y_binary):
    X, y = X_y_binary

    clf = LightGBMClassifier(n_jobs=1)
    sk_clf = lgbm.sklearn.LGBMClassifier(random_state=0, n_jobs=1)
    sk_clf.fit(X, y)
    sk_feature_importance = sk_clf.feature_importances_

    clf.fit(X, y)
    feature_importance = clf.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)


def test_fit_string_features(X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X['string_col'] = 'abc'

    # lightGBM requires input args to be int, float, or bool, not string
    X_expected = X.copy()
    X_expected['string_col'] = 0.0

    clf = lgbm.sklearn.LGBMClassifier(random_state=0, n_jobs=1)
    clf.fit(X_expected, y, categorical_feature=['string_col'])
    y_pred_sk = clf.predict(X_expected)
    y_pred_proba_sk = clf.predict_proba(X_expected)

    clf = LightGBMClassifier(n_jobs=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict_proba')
@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_fit_no_categories(mock_fit, mock_predict, mock_predict_proba, X_y_binary):
    X, y = X_y_binary
    X2 = pd.DataFrame(X)
    X2.columns = np.arange(len(X2.columns))
    clf = LightGBMClassifier(n_jobs=1)
    clf.fit(X, y)
    arg_X = mock_fit.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X2)

    clf.predict(X)
    arg_X = mock_predict.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X2)

    clf.predict_proba(X)
    arg_X = mock_predict_proba.call_args[0][0]
    np.testing.assert_array_equal(arg_X, X2)


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict_proba')
@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_correct_args(mock_fit, mock_predict, mock_predict_proba, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)

    # add object (string) and categorical data.
    X['string_col'] = 'abc'
    X['string_col'].iloc[len(X) // 2:] = 'cba'
    X['categorical_data'] = 'square'
    X['categorical_data'].iloc[len(X) // 2:] = 'circle'
    X['categorical_data'] = X['categorical_data'].astype('category')

    # create the expected result, which is a dataframe with int values in the categorical column and dtype=category
    X_expected = X.copy()
    X_expected = X_expected.replace(["abc", "cba"], [0.0, 1.0])
    X_expected = X_expected.replace(["square", "circle"], [1.0, 0.0])
    X_expected[['string_col', 'categorical_data']] = X_expected[['string_col', 'categorical_data']].astype('category')

    # rename the columns to be the indices
    X_expected.columns = np.arange(X_expected.shape[1])

    clf = LightGBMClassifier()
    clf.fit(X, y)
    arg_X = mock_fit.call_args[0][0]
    assert_frame_equal(X_expected, arg_X)

    clf.predict(X)
    arg_X = mock_predict.call_args[0][0]
    assert_frame_equal(X_expected, arg_X)

    clf.predict_proba(X)
    arg_X = mock_predict_proba.call_args[0][0]
    assert_frame_equal(X_expected, arg_X)


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict_proba')
@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_categorical_data_subset(mock_fit, mock_predict, mock_predict_proba, X_y_binary):
    X = pd.DataFrame({"feature_1": [0, 0, 1, 1, 0, 1], "feature_2": ["a", "a", "b", "b", "c", "c"]})
    y = pd.Series([1, 1, 0, 0, 0, 1])
    X_expected = pd.DataFrame({0: [0, 0, 1, 1, 0, 1], 1: [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]})
    X_expected.iloc[:, 1] = X_expected.iloc[:, 1].astype('category')

    X_subset = pd.DataFrame({"feature_1": [1, 0], "feature_2": ["c", "a"]})
    X_expected_subset = pd.DataFrame({0: [1, 0], 1: [2.0, 0.0]})
    X_expected_subset.iloc[:, 1] = X_expected_subset.iloc[:, 1].astype('category')

    clf = LightGBMClassifier()
    clf.fit(X, y)
    arg_X = mock_fit.call_args[0][0]
    assert_frame_equal(X_expected, arg_X)

    # determine whether predict and predict_proba perform as expected with the subset of categorical data
    clf.predict(X_subset)
    arg_X = mock_predict.call_args[0][0]
    assert_frame_equal(X_expected_subset, arg_X)

    clf.predict_proba(X_subset)
    arg_X = mock_predict_proba.call_args[0][0]
    assert_frame_equal(X_expected_subset, arg_X)


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict_proba')
@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_multiple_fit(mock_fit, mock_predict, mock_predict_proba):
    y = pd.Series([1] * 4)
    X1_fit = pd.DataFrame({"feature": ["a", "b", "c", "c"]})
    X1_fit_expected = pd.DataFrame({0: [0.0, 1.0, 2.0, 2.0]}, dtype='category')
    X1_predict = pd.DataFrame({"feature": ["a", "a", "b", "c"]})
    X1_predict_expected = pd.DataFrame({0: [0.0, 0.0, 1.0, 2.0]}, dtype='category')

    clf = LightGBMClassifier()
    clf.fit(X1_fit, y)
    assert_frame_equal(X1_fit_expected, mock_fit.call_args[0][0])
    clf.predict(X1_predict)
    assert_frame_equal(X1_predict_expected, mock_predict.call_args[0][0])
    clf.predict_proba(X1_predict)
    assert_frame_equal(X1_predict_expected, mock_predict_proba.call_args[0][0])

    # Check if it will fit a different dataset with new variable
    X2_fit = pd.DataFrame({"feature": ["c", "b", "a", "d"]})
    X2_fit_expected = pd.DataFrame({0: [2.0, 1.0, 0.0, 3.0]}, dtype='category')
    X2_predict = pd.DataFrame({"feature": ["d", "c", "b", "a"]})
    X2_predict_expected = pd.DataFrame({0: [3.0, 2.0, 1.0, 0.0]}, dtype='category')

    clf = LightGBMClassifier()
    clf.fit(X2_fit, y)
    assert_frame_equal(X2_fit_expected, mock_fit.call_args[0][0])
    clf.predict(X2_predict)
    assert_frame_equal(X2_predict_expected, mock_predict.call_args[0][0])
    clf.predict_proba(X2_predict)
    assert_frame_equal(X2_predict_expected, mock_predict_proba.call_args[0][0])


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_multiclass_label(mock_fit, mock_predict, X_y_multi):
    X, y = X_y_multi
    y_numeric = pd.Series(y, dtype='int64')
    y_alpha = pd.Series(y_numeric.copy().replace({0: "alright", 1: "better", 2: "great"}))

    clf = LightGBMClassifier()
    clf.fit(X, y_alpha)
    y_arg = mock_fit.call_args[0][1]
    assert_series_equal(y_arg, y_numeric)

    clf.predict(X)


@patch('evalml.pipelines.components.estimators.estimator.Estimator.predict')
@patch('evalml.pipelines.components.component_base.ComponentBase.fit')
def test_binary_label_encoding(mock_fit, mock_predict, X_y_binary):
    X, y = X_y_binary
    y_numeric = pd.Series(y, dtype='int64')
    y_alpha = pd.Series(y_numeric.copy().replace({0: "no", 1: "yes"}))

    clf = LightGBMClassifier()
    clf.fit(X, y_alpha)
    y_arg = mock_fit.call_args[0][1]
    assert_series_equal(y_arg, y_numeric)

    clf.predict(X)

    y_float = pd.Series(y_numeric.copy().replace({0: 0.99, 1: 1.01}))
    clf.fit(X, y_float)
    y_arg = mock_fit.call_args[0][1]
    assert_series_equal(y_arg, y_numeric)

    clf.predict(X)


def test_binary_rf(X_y_binary):
    X, y = X_y_binary

    with pytest.raises(lgbm.basic.LightGBMError, match="bagging_fraction"):
        clf = LightGBMClassifier(boosting_type="rf", bagging_freq=1, bagging_fraction=1.01)
        clf.fit(X, y)

    clf = LightGBMClassifier(boosting_type="rf", bagging_freq=0, n_jobs=1)
    clf.fit(X, y)
    assert clf.parameters['bagging_freq'] == 0
    assert clf.parameters['bagging_fraction'] == 0.9


def test_binary_goss(X_y_binary):
    X, y = X_y_binary
    clf = LightGBMClassifier(boosting_type="goss")
    clf.fit(X, y)
    assert clf.parameters['bagging_freq'] == 0
    assert clf.parameters['bagging_fraction'] == 0.9
