from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines import RegressionPipeline
from evalml.pipelines.components import RandomForestRegressor
from evalml.pipelines.components.ensemble import StackedEnsembleRegressor
from evalml.problem_types import ProblemTypes


def test_stacked_model_family():
    assert StackedEnsembleRegressor.model_family == ModelFamily.ENSEMBLE


def test_stacked_default_parameters():
    assert StackedEnsembleRegressor.default_parameters == {
        "final_estimator": StackedEnsembleRegressor._default_final_estimator,
        "n_jobs": -1,
    }


def test_stacked_ensemble_init_with_final_estimator(X_y_binary):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_binary
    component_graph = {
        "Random Forest": [RandomForestRegressor, "X", "y"],
        "Random Forest B": [RandomForestRegressor, "X", "y"],
        "Stacked Ensemble": [
            StackedEnsembleRegressor(n_jobs=1, final_estimator=RandomForestRegressor()),
            "Random Forest.x",
            "Random Forest B.x",
            "y",
        ],
    }
    pl = RegressionPipeline(component_graph)
    pl.fit(X, y)
    y_pred = pl.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_ensemble_does_not_overwrite_pipeline_random_seed():
    component_graph = {
        "Random Forest": [RandomForestRegressor(random_seed=3), "X", "y"],
        "Random Forest B": [RandomForestRegressor(random_seed=4), "X", "y"],
        "Stacked Ensemble": [
            StackedEnsembleRegressor(n_jobs=1, final_estimator=RandomForestRegressor()),
            "Random Forest.x",
            "Random Forest B.x",
            "y",
        ],
    }
    pl = RegressionPipeline(component_graph, random_seed=5)
    assert pl.random_seed == 5
    assert pl.get_component("Random Forest").random_seed == 3
    assert pl.get_component("Random Forest B").random_seed == 4


def test_stacked_problem_types():
    assert ProblemTypes.REGRESSION in StackedEnsembleRegressor.supported_problem_types
    assert len(StackedEnsembleRegressor.supported_problem_types) == 2


def test_stacked_fit_predict_classification(
    X_y_regression,
    stackable_regressors,
):
    def make_stacked_pipeline():
        component_graph = {}
        stacked_input = []
        for regressor in stackable_regressors:
            clf = regressor()
            component_graph[regressor.name] = [clf, "X", "y"]
            stacked_input.append(f"{regressor.name}.x")
        stacked_input.append("y")
        component_graph["Stacked Ensembler"] = [
            StackedEnsembleRegressor
        ] + stacked_input
        return RegressionPipeline(component_graph)

    X, y = X_y_regression
    pl = make_stacked_pipeline()
    pl.fit(X, y)
    y_pred = pl.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()

    pl = make_stacked_pipeline()
    pl.component_graph[-1].final_estimator = RandomForestRegressor()

    pl.fit(X, y)
    y_pred = pl.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()


@patch("evalml.pipelines.components.ensemble.StackedEnsembleRegressor.fit")
def test_stacked_feature_importance(mock_fit, X_y_regression):
    X, y = X_y_regression
    clf = StackedEnsembleRegressor(n_jobs=1)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(
        NotImplementedError,
        match="feature_importance is not implemented for StackedEnsembleClassifier and StackedEnsembleRegressor",
    ):
        clf.feature_importance
