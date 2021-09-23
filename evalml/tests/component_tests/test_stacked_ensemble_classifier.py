from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets

from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
)
from evalml.pipelines.components import (
    ElasticNetClassifier,
    Imputer,
    LogTransformer,
    RandomForestClassifier,
    StackedEnsembleClassifier,
)
from evalml.pipelines.utils import _make_stacked_ensemble_pipeline
from evalml.problem_types import ProblemTypes


def test_stacked_model_family():
    assert StackedEnsembleClassifier.model_family == ModelFamily.ENSEMBLE


def test_stacked_default_parameters():
    assert StackedEnsembleClassifier.default_parameters == {
        "final_estimator": StackedEnsembleClassifier._default_final_estimator,
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

    reg_pl_1 = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Log Transformer": ["Log Transformer", "X", "y"],
            "RF": ["Random Forest Classifier", "Imputer.x", "Log Transformer.y"],
        }
    )
    reg_pl_2 = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Log Transformer": ["Log Transformer", "X", "y"],
            "EN": ["Elastic Net Classifier", "Imputer.x", "Log Transformer.y"],
        }
    )

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.BINARY,
        use_sklearn=False,
    )
    check_for_components(ensemble_pipeline)

    reg_pl_1 = BinaryClassificationPipeline(
        {
            "Imputer": [Imputer, "X", "y"],
            "Log Transformer": [LogTransformer, "X", "y"],
            "RF": [RandomForestClassifier, "Imputer.x", "Log Transformer.y"],
        }
    )
    reg_pl_2 = BinaryClassificationPipeline(
        {
            "Imputer": [Imputer, "X", "y"],
            "Log Transformer": [LogTransformer, "X", "y"],
            "EN": [ElasticNetClassifier, "Imputer.x", "Log Transformer.y"],
        }
    )

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.REGRESSION,
        use_sklearn=False,
    )
    check_for_components(ensemble_pipeline)


def test_stacked_ensemble_nondefault_y():
    pytest.importorskip(
        "imblearn.over_sampling",
        reason="Skipping nondefault y test because imblearn not installed",
    )
    X, y = datasets.make_classification(
        n_samples=100, n_features=20, weights={0: 0.1, 1: 0.9}, random_state=0
    )
    input_pipelines = [
        BinaryClassificationPipeline(
            {
                "OS": ["Oversampler", "X", "y"],
                "rf": [RandomForestClassifier, "OS.x", "OS.y"],
            },
            parameters={"OS": {"sampling_ratio": 0.5}},
        ),
        BinaryClassificationPipeline(
            {
                "OS": ["Oversampler", "X", "y"],
                "rf": [RandomForestClassifier, "OS.x", "OS.y"],
            },
            parameters={
                "OS": {"sampling_ratio": 0.5},
                "Random Forest Classifier": {"n_estimators": 22},
            },
        ),
    ]

    pl = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.BINARY,
    )
    pl.fit(X, y)
    y_pred = pl.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_ensemble_keep_estimator_parameters(X_y_binary):
    X, y = X_y_binary
    input_pipelines = [
        BinaryClassificationPipeline(
            {
                "Impute": [Imputer, "X", "y"],
                "Random Forest Classifier": [RandomForestClassifier, "Impute.x", "y"],
            },
            parameters={"Impute": {"numeric_fill_value": 10}},
        ),
        BinaryClassificationPipeline(
            [RandomForestClassifier],
            parameters={"Random Forest Classifier": {"n_estimators": 22}},
        ),
    ]

    pl = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.BINARY,
    )
    assert (
        pl.get_component("Random Forest Pipeline - Impute").parameters[
            "numeric_fill_value"
        ]
        == 10
    )
    assert (
        pl.get_component(
            "Random Forest Pipeline 2 - Random Forest Classifier"
        ).parameters["n_estimators"]
        == 22
    )


@patch("evalml.pipelines.components.StackedEnsembleClassifier.fit")
@patch("evalml.pipelines.components.ElasticNetClassifier.fit")
@patch("evalml.pipelines.components.RandomForestClassifier.fit")
@patch("evalml.pipelines.components.ElasticNetClassifier.predict_proba")
@patch("evalml.pipelines.components.RandomForestClassifier.predict_proba")
def test_ensembler_use_component_preds_binary(
    mock_rf_predict_proba,
    mock_en_predict_proba,
    mock_rf_fit,
    mock_en_fit,
    mock_ensembler,
    X_y_binary,
):
    X, y = X_y_binary

    mock_en_predict_proba_series = pd.Series(np.zeros(len(y)))
    mock_en_predict_proba_series.ww.init()
    mock_en_predict_proba.return_value = mock_en_predict_proba_series

    mock_rf_predict_proba_series = pd.Series(np.ones(len(y)))
    mock_rf_predict_proba_series.ww.init()
    mock_rf_predict_proba.return_value = mock_rf_predict_proba_series

    reg_pl_1 = BinaryClassificationPipeline([RandomForestClassifier])
    reg_pl_2 = BinaryClassificationPipeline([ElasticNetClassifier])

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.BINARY,
        use_sklearn=False,
    )
    ensemble_pipeline.fit(X, y)
    ensemble_input, _ = mock_ensembler.call_args[0]

    assert ensemble_input.shape == (100, 2)
    assert ensemble_input["Linear Pipeline - Elastic Net Classifier.x"].equals(
        pd.Series(np.zeros(len(y)))
    )
    assert ensemble_input["Random Forest Pipeline - Random Forest Classifier.x"].equals(
        pd.Series(np.ones(len(y)))
    )


@patch("evalml.pipelines.components.StackedEnsembleClassifier.fit")
@patch("evalml.pipelines.components.ElasticNetClassifier.fit")
@patch("evalml.pipelines.components.RandomForestClassifier.fit")
@patch("evalml.pipelines.components.ElasticNetClassifier.predict_proba")
@patch("evalml.pipelines.components.RandomForestClassifier.predict_proba")
def test_ensembler_use_component_preds_multi(
    mock_rf_predict_proba,
    mock_en_predict_proba,
    mock_rf_fit,
    mock_en_fit,
    mock_ensembler,
    X_y_multi,
):
    X, y = X_y_multi

    mock_en_predict_proba_df = pd.DataFrame(np.zeros((len(y), 3)))
    mock_en_predict_proba_df.ww.init()
    mock_en_predict_proba.return_value = mock_en_predict_proba_df

    mock_rf_predict_proba_df = pd.DataFrame(np.ones((len(y), 3)))
    mock_rf_predict_proba_df.ww.init()
    mock_rf_predict_proba.return_value = mock_rf_predict_proba_df

    reg_pl_1 = MulticlassClassificationPipeline([RandomForestClassifier])
    reg_pl_2 = MulticlassClassificationPipeline([ElasticNetClassifier])

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.MULTICLASS,
        use_sklearn=False,
    )
    ensemble_pipeline.fit(X, y)
    ensemble_input, _ = mock_ensembler.call_args[0]

    assert ensemble_input.shape == (100, 6)
    for i in range(0, 3):
        assert ensemble_input[
            f"Col {i} Linear Pipeline - Elastic Net Classifier.x"
        ].equals(pd.Series(np.zeros(len(y))))
        assert ensemble_input[
            f"Col {i} Random Forest Pipeline - Random Forest Classifier.x"
        ].equals(pd.Series(np.ones(len(y))))
