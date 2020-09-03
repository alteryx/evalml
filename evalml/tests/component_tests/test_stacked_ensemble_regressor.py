
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components.ensemble import StackedEnsembleRegressor
from evalml.problem_types import ProblemTypes


@pytest.fixture
def stackable_regressors(all_regression_estimators_classes):
    estimators = [estimator_class() for estimator_class in all_regression_estimators_classes if estimator_class.model_family != ModelFamily.ENSEMBLE and estimator_class.model_family != ModelFamily.BASELINE]
    return estimators


def test_stacked_model_family():
    assert StackedEnsembleRegressor.model_family == ModelFamily.ENSEMBLE


def test_stacked_ensemble_parameters(stackable_regressors):
    clf = StackedEnsembleRegressor(stackable_regressors, final_estimator=None, random_state=2)
    expected_parameters = {
        "estimators": stackable_regressors,
        "final_estimator": None,
        'cv': None,
        'n_jobs': -1
    }
    assert clf.parameters == expected_parameters


def test_stacked_problem_types():
    assert ProblemTypes.REGRESSION in StackedEnsembleRegressor.supported_problem_types
    assert len(StackedEnsembleRegressor.supported_problem_types) == 1


def test_stacked_fit_predict_regression(X_y_regression, stackable_regressors):
    X, y = X_y_regression
    clf = StackedEnsembleRegressor(stackable_regressors, final_estimator=None, random_state=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)


def test_stacked_feature_importance(X_y_regression, stackable_regressors):
    X, y = X_y_regression
    clf = StackedEnsembleRegressor(stackable_regressors, final_estimator=None, random_state=2)
    clf.fit(X, y)
    clf.feature_importance