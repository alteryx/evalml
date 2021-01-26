import string

import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components import Estimator
from evalml.pipelines.components.utils import (
    _all_estimators_used_in_search,
    get_estimators
)
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_random_state


def test_estimators_feature_name_with_random_ascii(X_y_binary, X_y_multi, X_y_regression, helper_functions):
    for estimator_class in _all_estimators_used_in_search():
        supported_problem_types = [handle_problem_types(pt) for pt in estimator_class.supported_problem_types]
        for problem_type in supported_problem_types:
            clf = helper_functions.safe_init_component_with_njobs_1(estimator_class)
            if problem_type == ProblemTypes.BINARY:
                X, y = X_y_binary
            elif problem_type == ProblemTypes.MULTICLASS:
                X, y = X_y_multi
            elif problem_type == ProblemTypes.REGRESSION:
                X, y = X_y_regression

            X = get_random_state(clf.random_state).random((X.shape[0], len(string.printable)))
            col_names = ['column_{}'.format(ascii_char) for ascii_char in string.printable]
            X = pd.DataFrame(X, columns=col_names)
            clf.fit(X, y)
            assert len(clf.feature_importance) == len(X.columns)
            assert not np.isnan(clf.feature_importance).all().all()
            predictions = clf.predict(X)
            assert len(predictions) == len(y)
            assert not np.isnan(predictions).all()


def test_binary_classification_estimators_predict_proba_col_order(helper_functions):
    X = pd.DataFrame({'input': np.concatenate([np.array([-1] * 100), np.array([1] * 100)])})
    data = np.concatenate([np.zeros(100), np.ones(100)])
    y = pd.Series(data)
    for estimator_class in _all_estimators_used_in_search():
        supported_problem_types = [handle_problem_types(pt) for pt in estimator_class.supported_problem_types]
        if ProblemTypes.BINARY in supported_problem_types:
            estimator = helper_functions.safe_init_component_with_njobs_1(estimator_class)
            estimator.fit(X, y)
            predicted_proba = estimator.predict_proba(X)
            expected = np.concatenate([(1 - data).reshape(-1, 1), data.reshape(-1, 1)], axis=1)
            np.testing.assert_allclose(expected, np.round(predicted_proba).values)


def test_estimator_equality_different_supported_problem_types():
    class MockEstimator(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ['binary']

    mock_estimator = MockEstimator()
    mock_estimator.supported_problem_types = ['binary', 'multiclass']
    assert mock_estimator != MockEstimator()
    assert 'Mock Estimator' != mock_estimator


@pytest.mark.parametrize("data_type", ['li', 'np', 'pd', 'ww'])
def test_all_estimators_check_fit_input_type(data_type, X_y_binary, make_data_type, helper_functions):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    estimators_to_check = [estimator for estimator in get_estimators('binary')]
    for component_class in estimators_to_check:
        component = helper_functions.safe_init_component_with_njobs_1(component_class)
        component.fit(X, y)
        component.predict(X)
        component.predict_proba(X)


@pytest.mark.parametrize("data_type", ['li', 'np', 'pd', 'ww'])
def test_all_estimators_check_fit_input_type_regression(data_type, X_y_regression, make_data_type, helper_functions):
    X, y = X_y_regression
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    estimators_to_check = [estimator for estimator in get_estimators('regression')]
    for component_class in estimators_to_check:
        component = helper_functions.safe_init_component_with_njobs_1(component_class)
        component.fit(X, y)
        component.predict(X)
