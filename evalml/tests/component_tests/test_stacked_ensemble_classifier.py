from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
)
from evalml.pipelines.components import RandomForestClassifier
from evalml.pipelines.components.ensemble import StackedEnsembleClassifier
from evalml.problem_types import ProblemTypes


def test_stacked_model_family():
    assert StackedEnsembleClassifier.model_family == ModelFamily.ENSEMBLE


def test_stacked_default_parameters():
    assert StackedEnsembleClassifier.default_parameters == {
        "final_estimator": None,
        "n_jobs": -1,
    }


def test_stacked_ensemble_init_with_final_estimator(X_y_binary):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_binary
    component_graph = {
        "Random Forest": [RandomForestClassifier, "X", "y"],
        "Random Forest B": [RandomForestClassifier, "X", "y"],
        "Stacked Ensemble": [
            StackedEnsembleClassifier(
                n_jobs=1, final_estimator=RandomForestClassifier()
            ),
            "Random Forest.x",
            "Random Forest B.x",
            "y",
        ],
    }
    pl = BinaryClassificationPipeline(component_graph)
    pl.fit(X, y)
    y_pred = pl.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_ensemble_does_not_overwrite_pipeline_random_seed():
    component_graph = {
        "Random Forest": [RandomForestClassifier(random_seed=3), "X", "y"],
        "Random Forest B": [RandomForestClassifier(random_seed=4), "X", "y"],
        "Stacked Ensemble": [
            StackedEnsembleClassifier(
                n_jobs=1, final_estimator=RandomForestClassifier()
            ),
            "Random Forest.x",
            "Random Forest B.x",
            "y",
        ],
    }
    pl = BinaryClassificationPipeline(component_graph, random_seed=5)
    assert pl.random_seed == 5
    assert pl.get_component("Random Forest").random_seed == 3
    assert pl.get_component("Random Forest B").random_seed == 4


def test_stacked_problem_types():
    assert ProblemTypes.BINARY in StackedEnsembleClassifier.supported_problem_types
    assert ProblemTypes.MULTICLASS in StackedEnsembleClassifier.supported_problem_types
    assert StackedEnsembleClassifier.supported_problem_types == [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_stacked_fit_predict_classification(
    X_y_binary, X_y_multi, stackable_classifiers, problem_type
):
    def make_stacked_pipeline(pipeline_class):
        component_graph = {}
        stacked_input = []
        for classifier in stackable_classifiers:
            clf = classifier()
            component_graph[classifier.name] = [clf, "X", "y"]
            stacked_input.append(f"{classifier.name}.x")
        stacked_input.append("y")
        component_graph["Stacked Ensembler"] = [
            StackedEnsembleClassifier
        ] + stacked_input
        return pipeline_class(component_graph)

    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        num_classes = 2
        pipeline_class = BinaryClassificationPipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        num_classes = 3
        pipeline_class = MulticlassClassificationPipeline

    pl = make_stacked_pipeline(pipeline_class)

    pl.fit(X, y)
    y_pred = pl.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()

    y_pred_proba = pl.predict_proba(X)
    assert isinstance(y_pred_proba, pd.DataFrame)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert not np.isnan(y_pred_proba).all().all()

    pl = make_stacked_pipeline(pipeline_class)
    pl.component_graph[-1].final_estimator = RandomForestClassifier()

    pl.fit(X, y)
    y_pred = pl.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()

    y_pred_proba = pl.predict_proba(X)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert isinstance(y_pred_proba, pd.DataFrame)
    assert not np.isnan(y_pred_proba).all().all()


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@patch("evalml.pipelines.components.ensemble.StackedEnsembleClassifier.fit")
def test_stacked_feature_importance(mock_fit, X_y_binary, X_y_multi, problem_type):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
    clf = StackedEnsembleClassifier(n_jobs=1)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(
        NotImplementedError,
        match="feature_importance is not implemented for StackedEnsembleClassifier and StackedEnsembleRegressor",
    ):
        clf.feature_importance
