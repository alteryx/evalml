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
                n_jobs=1,
                final_estimator=RandomForestClassifier(),
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
                n_jobs=1,
                final_estimator=RandomForestClassifier(),
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
    X_y_binary,
    X_y_multi,
    stackable_classifiers,
    problem_type,
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
            StackedEnsembleClassifier,
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
        },
    )
    reg_pl_2 = BinaryClassificationPipeline(
        {
            "Imputer": ["Imputer", "X", "y"],
            "Log Transformer": ["Log Transformer", "X", "y"],
            "EN": ["Elastic Net Classifier", "Imputer.x", "Log Transformer.y"],
        },
    )

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.BINARY,
    )
    check_for_components(ensemble_pipeline)

    reg_pl_1 = BinaryClassificationPipeline(
        {
            "Imputer": [Imputer, "X", "y"],
            "Log Transformer": [LogTransformer, "X", "y"],
            "RF": [RandomForestClassifier, "Imputer.x", "Log Transformer.y"],
        },
    )
    reg_pl_2 = BinaryClassificationPipeline(
        {
            "Imputer": [Imputer, "X", "y"],
            "Log Transformer": [LogTransformer, "X", "y"],
            "EN": [ElasticNetClassifier, "Imputer.x", "Log Transformer.y"],
        },
    )

    ensemble_pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=[reg_pl_1, reg_pl_2],
        problem_type=ProblemTypes.REGRESSION,
    )
    check_for_components(ensemble_pipeline)


def test_stacked_ensemble_nondefault_y():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=20,
        weights={0: 0.1, 1: 0.9},
        random_state=0,
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
            "Random Forest Pipeline 2 - Random Forest Classifier",
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
    )
    ensemble_pipeline.fit(X, y)
    ensemble_input, _ = mock_ensembler.call_args[0]

    assert ensemble_input.shape == (100, 2)
    assert ensemble_input["Linear Pipeline - Elastic Net Classifier.x"].equals(
        pd.Series(np.zeros(len(y))),
    )
    assert ensemble_input["Random Forest Pipeline - Random Forest Classifier.x"].equals(
        pd.Series(np.ones(len(y))),
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


def test_stacked_ensemble_cache(X_y_binary):
    X, y = X_y_binary
    trained_imputer = Imputer()
    trained_imputer.fit(X, y)
    trained_rf = RandomForestClassifier()
    trained_rf.fit(X, y)
    trained_en = ElasticNetClassifier()
    trained_en.fit(X, y)
    cache = {
        ModelFamily.RANDOM_FOREST: {
            "random_hash": {
                "Impute": trained_imputer,
                "Random Forest Classifier": trained_rf,
            },
        },
        ModelFamily.LINEAR_MODEL: {
            "random_hash": {"Elastic Net Classifier": trained_en},
        },
    }
    input_pipelines = [
        BinaryClassificationPipeline(
            {
                "Impute": [Imputer, "X", "y"],
                "Random Forest Classifier": [RandomForestClassifier, "Impute.x", "y"],
            },
        ),
        BinaryClassificationPipeline(
            {"Elastic Net Classifier": [ElasticNetClassifier, "X", "y"]},
        ),
    ]

    expected_cached_data = {
        "random_hash": {
            "Random Forest Pipeline - Impute": trained_imputer,
            "Random Forest Pipeline - Random Forest Classifier": trained_rf,
            "Linear Pipeline - Elastic Net Classifier": trained_en,
        },
    }

    pl = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.BINARY,
        cached_data=cache,
    )
    assert pl.component_graph.cached_data == expected_cached_data


@patch(
    "evalml.pipelines.component_graph.ComponentGraph._consolidate_inputs_for_component",
)
@patch(
    "evalml.pipelines.components.transformers.encoders.label_encoder.LabelEncoder.transform",
)
@patch("evalml.pipelines.components.transformers.imputers.imputer.Imputer.fit")
@patch("evalml.pipelines.components.estimators.Estimator.fit")
@patch("evalml.pipelines.components.transformers.imputers.imputer.Imputer.transform")
@patch("evalml.pipelines.components.estimators.Estimator.predict")
@patch("evalml.pipelines.components.estimators.Estimator.predict_proba")
def test_stacked_ensemble_cache_training(
    mock_estimator_predict_proba,
    mock_estimator_predict,
    mock_transformer_transform,
    mock_estimator_fit,
    mock_transform_fit,
    mock_label,
    mock_consolidate,
    X_y_binary,
):
    X, y = X_y_binary
    mock_estimator_predict.return_value = y
    mock_transformer_transform.return_value = X
    mock_label.return_value = y

    trained_imputer = Imputer()
    trained_rf = RandomForestClassifier()
    # make the components 'trained'
    trained_imputer._is_fitted = True
    trained_rf._is_fitted = True
    hashes = hash(tuple(X.index))
    cache = {
        ModelFamily.RANDOM_FOREST: {
            hashes: {"Impute": trained_imputer, "Random Forest Classifier": trained_rf},
        },
    }

    input_pipelines = [
        BinaryClassificationPipeline(
            {
                "Impute": [Imputer, "X", "y"],
                "Random Forest Classifier": [RandomForestClassifier, "Impute.x", "y"],
            },
        ),
    ]

    mock_consolidate.return_value = (X, y)

    pl_cache = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.BINARY,
        cached_data=cache,
    )
    pl_no_cache = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.BINARY,
        cached_data=None,
    )

    # ensure we are not calling transform or fit for the components we have in our cache
    pl_cache.fit(X, y)
    mock_transform_fit.assert_not_called()
    # ensure the only estimator fit call that happens is the final metalearner estimator
    mock_estimator_fit.assert_called_once()

    mock_estimator_fit.reset_mock()
    mock_transform_fit.reset_mock()

    # ensure if we remove the cache, we do train the components appropriately
    predicted = pd.DataFrame(
        {
            "0": [0.9 if i == 1 else 0.1 for i in y],
            "1": [0.1 if i == 1 else 0.9 for i in y],
        },
    )
    predicted.ww.init()
    mock_estimator_predict_proba.return_value = predicted
    pl_no_cache.fit(X, y)
    mock_transform_fit.assert_called_once()
    assert mock_estimator_fit.call_count == 2


@pytest.mark.parametrize("indices", [0, 1])
def test_stacked_ensemble_cache_train_predict(
    indices,
    X_y_binary,
):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X2 = X.sample(frac=1)

    trained_imputer = Imputer()
    trained_rf = RandomForestClassifier()
    trained_imputer.fit(X2, y)
    trained_rf.fit(X2, y)
    if indices == 0:
        hashes = hash(tuple(X2.index))
    else:
        hashes = hash(tuple(X.index))
    cache = {
        ModelFamily.RANDOM_FOREST: {
            hashes: {"Impute": trained_imputer, "Random Forest Classifier": trained_rf},
        },
    }

    input_pipelines = [
        BinaryClassificationPipeline(
            {
                "Impute": [Imputer, "X", "y"],
                "Random Forest Classifier": [RandomForestClassifier, "Impute.x", "y"],
            },
        ),
    ]

    pl_cache = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=ProblemTypes.BINARY,
        cached_data=cache,
    )
    pl_cache_copy = pl_cache.clone()
    pl_cache.fit(X2, y)
    pl_cache_copy.fit(X2, y)

    try:
        pd.testing.assert_frame_equal(
            pl_cache.predict_proba(X2),
            pl_cache_copy.predict_proba(X2),
        )
        assert indices == 0
    except AssertionError:
        assert indices == 1
