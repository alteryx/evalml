from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines import RegressionPipeline
from evalml.pipelines.components import (
    ElasticNetRegressor,
    Imputer,
    LogTransformer,
    RandomForestRegressor,
)
from evalml.pipelines.components.ensemble import StackedEnsembleRegressor
from evalml.pipelines.utils import _make_stacked_ensemble_pipeline
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
            StackedEnsembleRegressor,
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


def test_stacked_same_model_family():
    graph_en = {
        "Imputer": ["Imputer", "X", "y"],
        "Target Imputer": ["Target Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "Target Imputer.y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Regressor", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Regressor", "OneHot_ElasticNet.x", "y"],
        "EN": [
            "Elastic Net Regressor",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }

    graph_linear = {
        "Imputer": ["Imputer", "X", "y"],
        "Target Imputer": ["Target Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "Target Imputer.y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Regressor", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Regressor", "OneHot_ElasticNet.x", "y"],
        "Linear Regressor": [
            "Linear Regressor",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }

    input_pipelines = [
        RegressionPipeline(component_graph=graph_en),
        RegressionPipeline(component_graph=graph_linear),
    ]

    pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.REGRESSION,
    )

    assert "Linear Pipeline - Imputer" in pipeline.component_graph.compute_order
    assert "Linear Pipeline 2 - Imputer" in pipeline.component_graph.compute_order

    assert "Linear Pipeline - Elastic Net" in pipeline.component_graph.compute_order
    assert "Linear Pipeline 2 - Elastic Net" in pipeline.component_graph.compute_order
    assert "Linear Pipeline - EN" in pipeline.component_graph.compute_order


def test_ensembler_str_and_classes():
    """
    Test that ensures that pipelines that are defined as strings or classes are able to be ensembled.
    """

    def check_for_components(pl):
        pl_components = pl.component_graph.compute_order
        expected_components = [
            "Linear Pipeline - Imputer",
            "Linear Pipeline - Log Transformer",
            "Linear Pipeline - EN",
            "Random Forest Pipeline - Imputer",
            "Random Forest Pipeline - Log Transformer",
            "Random Forest Pipeline - RF",
        ]
        for component in expected_components:
            assert component in pl_components

    reg_pl_1 = RegressionPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Log Transformer": ["Log Transformer", "X", "y"],
            "RF": ["Random Forest Regressor", "Imputer.x", "Log Transformer.y"],
        },
    )
    reg_pl_2 = RegressionPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Log Transformer": ["Log Transformer", "X", "y"],
            "EN": ["Elastic Net Regressor", "Imputer.x", "Log Transformer.y"],
        },
    )

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.REGRESSION,
    )
    check_for_components(ensemble_pipeline)

    reg_pl_1 = RegressionPipeline(
        {
            "Imputer": [Imputer, "X", "y"],
            "Log Transformer": [LogTransformer, "X", "y"],
            "RF": [RandomForestRegressor, "Imputer.x", "Log Transformer.y"],
        },
    )
    reg_pl_2 = RegressionPipeline(
        {
            "Imputer": [Imputer, "X", "y"],
            "Log Transformer": [LogTransformer, "X", "y"],
            "EN": [ElasticNetRegressor, "Imputer.x", "Log Transformer.y"],
        },
    )

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.REGRESSION,
    )
    check_for_components(ensemble_pipeline)


@patch("evalml.pipelines.components.StackedEnsembleRegressor.fit")
@patch("evalml.pipelines.components.ElasticNetRegressor.fit")
@patch("evalml.pipelines.components.RandomForestRegressor.fit")
@patch("evalml.pipelines.components.ElasticNetRegressor.predict_proba")
@patch("evalml.pipelines.components.RandomForestRegressor.predict_proba")
def test_ensembler_use_component_preds(
    mock_rf_predict_proba,
    mock_en_predict_proba,
    mock_rf_fit,
    mock_en_fit,
    mock_ensembler,
    X_y_regression,
):
    X, y = X_y_regression

    mock_en_predict_proba_series = pd.Series(np.zeros(len(y)))
    mock_en_predict_proba_series.ww.init()
    mock_en_predict_proba.return_value = mock_en_predict_proba_series

    mock_rf_predict_proba_series = pd.Series(np.ones(len(y)))
    mock_rf_predict_proba_series.ww.init()
    mock_rf_predict_proba.return_value = mock_rf_predict_proba_series

    reg_pl_1 = RegressionPipeline([RandomForestRegressor])
    reg_pl_2 = RegressionPipeline([ElasticNetRegressor])

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.REGRESSION,
    )
    ensemble_pipeline.fit(X, y)
    ensemble_input, _ = mock_ensembler.call_args[0]

    assert ensemble_input.shape == (100, 2)
    assert ensemble_input["Linear Pipeline - Elastic Net Regressor.x"].equals(
        pd.Series(np.zeros(len(y))),
    )
    assert ensemble_input["Random Forest Pipeline - Random Forest Regressor.x"].equals(
        pd.Series(np.ones(len(y))),
    )


def test_stacked_ensemble_cache_train_predict(
    X_y_binary,
):
    X, y = X_y_binary
    X = pd.DataFrame(X)

    trained_imputer = Imputer()
    trained_rf = RandomForestRegressor()
    trained_imputer.fit(X, y)
    trained_rf.fit(X, y)
    hashes = hash(tuple(X.index))
    cache = {
        ModelFamily.RANDOM_FOREST: {
            hashes: {"Impute": trained_imputer, "Random Forest Regressor": trained_rf},
        },
    }

    input_pipelines = [
        RegressionPipeline(
            {
                "Impute": [Imputer, "X", "y"],
                "Random Forest Regressor": [RandomForestRegressor, "Impute.x", "y"],
            },
        ),
    ]

    pl_cache = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.REGRESSION,
        cached_data=cache,
    )
    pl_no_cache = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.REGRESSION,
        cached_data=None,
    )
    pl_cache.fit(X, y)
    pl_no_cache.fit(X, y)

    pd.testing.assert_series_equal(pl_cache.predict(X), pl_no_cache.predict(X))
