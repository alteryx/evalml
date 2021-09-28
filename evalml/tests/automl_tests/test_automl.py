import os
import warnings
from collections import OrderedDict, defaultdict
from itertools import product
from unittest.mock import MagicMock, PropertyMock, patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from joblib import hash as joblib_hash
from sklearn.model_selection import KFold, StratifiedKFold
from skopt.space import Categorical, Integer, Real

from evalml import AutoMLSearch
from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.automl.automl_search import build_engine_from_str
from evalml.automl.callbacks import (
    log_error_callback,
    raise_error_callback,
    silent_error_callback,
)
from evalml.automl.engine import CFEngine, DaskEngine, SequentialEngine
from evalml.automl.utils import (
    _LARGE_DATA_PERCENT_VALIDATION,
    _LARGE_DATA_ROW_THRESHOLD,
    get_default_primary_search_objective,
    get_pipelines_from_component_graphs,
)
from evalml.exceptions import (
    AutoMLSearchException,
    ParameterNotUsedWarning,
    PipelineNotFoundError,
    PipelineNotYetFittedError,
    PipelineScoreError,
)
from evalml.model_family import ModelFamily
from evalml.objectives import (
    F1,
    BinaryClassificationObjective,
    CostBenefitMatrix,
    FraudCost,
    RegressionObjective,
)
from evalml.objectives.utils import (
    get_all_objective_names,
    get_core_objectives,
    get_non_core_objectives,
    get_objective,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    ComponentGraph,
    Estimator,
    MulticlassClassificationPipeline,
    PipelineBase,
    RegressionPipeline,
    SklearnStackedEnsembleClassifier,
)
from evalml.pipelines.components.utils import (
    allowed_model_families,
    get_estimators,
)
from evalml.pipelines.utils import make_pipeline
from evalml.preprocessing import TrainingValidationSplit
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_classification,
    is_time_series,
)
from evalml.tests.automl_tests.parallel_tests.test_automl_dask import (
    engine_strs,
)
from evalml.tests.conftest import CustomClassificationObjectiveRanges
from evalml.tuners import NoParamsException, RandomSearchTuner, SKOptTuner


@pytest.mark.parametrize(
    "automl_type,objective",
    zip(
        [
            ProblemTypes.REGRESSION,
            ProblemTypes.MULTICLASS,
            ProblemTypes.BINARY,
            ProblemTypes.BINARY,
        ],
        ["R2", "log loss multiclass", "log loss binary", "F1"],
    ),
)
def test_search_results(X_y_regression, X_y_binary, X_y_multi, automl_type, objective):
    expected_cv_data_keys = {
        "all_objective_scores",
        "mean_cv_score",
        "binary_classification_threshold",
    }
    if automl_type == ProblemTypes.REGRESSION:
        expected_pipeline_class = RegressionPipeline
        X, y = X_y_regression
    elif automl_type == ProblemTypes.BINARY:
        expected_pipeline_class = BinaryClassificationPipeline
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        expected_pipeline_class = MulticlassClassificationPipeline
        X, y = X_y_multi

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        objective=objective,
        max_iterations=2,
        n_jobs=1,
    )
    automl.search()
    assert automl.results.keys() == {"pipeline_results", "search_order"}
    assert automl.results["search_order"] == [0, 1]
    assert len(automl.results["pipeline_results"]) == 2
    for pipeline_id, results in automl.results["pipeline_results"].items():
        assert results.keys() == {
            "id",
            "pipeline_name",
            "pipeline_class",
            "pipeline_summary",
            "parameters",
            "mean_cv_score",
            "standard_deviation_cv_score",
            "high_variance_cv",
            "training_time",
            "cv_data",
            "percent_better_than_baseline_all_objectives",
            "percent_better_than_baseline",
            "validation_score",
        }
        assert results["id"] == pipeline_id
        assert isinstance(results["pipeline_name"], str)
        assert issubclass(results["pipeline_class"], expected_pipeline_class)
        assert isinstance(results["pipeline_summary"], str)
        assert isinstance(results["parameters"], dict)
        assert isinstance(results["mean_cv_score"], float)
        assert isinstance(results["high_variance_cv"], bool)
        assert isinstance(results["cv_data"], list)
        for cv_result in results["cv_data"]:
            assert cv_result.keys() == expected_cv_data_keys
            if objective == "F1":
                assert cv_result["binary_classification_threshold"] == 0.5
            else:
                assert cv_result["binary_classification_threshold"] is None
            all_objective_scores = cv_result["all_objective_scores"]
            for score in all_objective_scores.values():
                assert score is not None
        assert automl.get_pipeline(pipeline_id).parameters == results["parameters"]
        assert (
            results["validation_score"]
            == pd.Series([fold["mean_cv_score"] for fold in results["cv_data"]])[0]
        )
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)
    assert np.all(
        automl.rankings.dtypes
        == pd.Series(
            [
                np.dtype("int64"),
                np.dtype("O"),
                np.dtype("int64"),
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("bool"),
                np.dtype("O"),
            ],
            index=[
                "id",
                "pipeline_name",
                "search_order",
                "mean_cv_score",
                "standard_deviation_cv_score",
                "validation_score",
                "percent_better_than_baseline",
                "high_variance_cv",
                "parameters",
            ],
        )
    )
    assert np.all(
        automl.full_rankings.dtypes
        == pd.Series(
            [
                np.dtype("int64"),
                np.dtype("O"),
                np.dtype("int64"),
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("float64"),
                np.dtype("bool"),
                np.dtype("O"),
            ],
            index=[
                "id",
                "pipeline_name",
                "search_order",
                "mean_cv_score",
                "standard_deviation_cv_score",
                "validation_score",
                "percent_better_than_baseline",
                "high_variance_cv",
                "parameters",
            ],
        )
    )


@pytest.mark.parametrize(
    "automl_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
@pytest.mark.parametrize("verbose", [True, False])
def test_pipeline_limits(
    automl_type,
    verbose,
    caplog,
    AutoMLTestEnv,
    X_y_binary,
    X_y_multi,
    X_y_regression,
):
    caplog.clear()
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        score_value = {"Log Loss Binary": 1.0}
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        score_value = {"Log Loss Multiclass": 1.0}
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        score_value = {"R2": 1.0}

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        max_iterations=1,
        verbose=verbose,
    )
    env = AutoMLTestEnv(automl_type)
    with env.test_context(score_return_value=score_value):
        automl.search()
    out = caplog.text
    assert ("Searching up to 1 pipelines. " in out) == verbose
    assert len(automl.results["pipeline_results"]) == 1

    caplog.clear()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        max_time=1,
        verbose=verbose,
    )
    with env.test_context(score_return_value=score_value):
        automl.search()
    out = caplog.text
    assert ("Will stop searching for new pipelines after 1 seconds" in out) == verbose
    assert len(automl.results["pipeline_results"]) >= 1

    caplog.clear()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        max_time=1,
        max_iterations=5,
        verbose=verbose,
    )
    with env.test_context(score_return_value=score_value):
        automl.search()
    out = caplog.text
    if verbose:
        assert "Searching up to 5 pipelines. " in out
        assert "Will stop searching for new pipelines after 1 seconds" in out
    else:
        assert "Searching up to 5 pipelines. " not in out
        assert "Will stop searching for new pipelines after 1 seconds" not in out
    assert len(automl.results["pipeline_results"]) <= 5

    caplog.clear()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        verbose=verbose,
    )
    with env.test_context(score_return_value=score_value):
        automl.search()
    out = caplog.text
    if verbose:
        assert "Using default limit of max_batches=1." in out
        assert "Searching up to 1 batches for a total of" in out
    else:
        assert "Using default limit of max_batches=1." not in out
        assert "Searching up to 1 batches for a total of" not in out
    assert len(automl.results["pipeline_results"]) > 4

    caplog.clear()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        max_time=1e-16,
        verbose=verbose,
    )
    with env.test_context(score_return_value=score_value):
        automl.search()
    out = caplog.text
    assert ("Will stop searching for new pipelines after 0 seconds" in out) == verbose
    # search will always run at least one pipeline
    assert len(automl.results["pipeline_results"]) >= 1
    caplog.clear()


def test_pipeline_fit_raises(AutoMLTestEnv, X_y_binary, caplog):
    X, y = X_y_binary
    # Don't train the best pipeline, since this test mocks the pipeline.fit() method and causes it to raise an exception,
    # which we don't want to raise while fitting the best pipeline.
    caplog.clear()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
        train_best_pipeline=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(
        mock_fit_side_effect=Exception("all your model are belong to us")
    ):
        automl.search()
    out = caplog.text
    assert "Exception during automl search" in out
    pipeline_results = automl.results.get("pipeline_results", {})
    assert len(pipeline_results) == 1

    cv_scores_all = pipeline_results[0].get("cv_data", {})
    for cv_scores in cv_scores_all:
        for name, score in cv_scores["all_objective_scores"].items():
            if name in ["# Training", "# Validation"]:
                assert score > 0
            else:
                assert np.isnan(score)


def test_pipeline_score_raises(AutoMLTestEnv, X_y_binary, caplog):
    caplog.clear()
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(
        mock_score_side_effect=Exception("all your model are belong to us")
    ):
        automl.search()
    out = caplog.text
    assert "Exception during automl search" in out
    assert "All scores will be replaced with nan." in out
    pipeline_results = automl.results.get("pipeline_results", {})
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0]["cv_data"][0]["all_objective_scores"]
    objective_scores = {
        o.name: cv_scores_all[o.name]
        for o in [automl.objective] + automl.additional_objectives
    }

    assert np.isnan(list(objective_scores.values())).all()


@patch("evalml.objectives.AUC.score")
def test_objective_score_raises(mock_score, X_y_binary, caplog):
    caplog.clear()
    msg = "all your model are belong to us"
    mock_score.side_effect = Exception(msg)
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=1,
        n_jobs=1,
    )
    automl.search()
    out = caplog.text

    assert msg in out
    pipeline_results = automl.results.get("pipeline_results")
    assert len(pipeline_results) == 1
    cv_scores_all = pipeline_results[0].get("cv_data")
    scores = cv_scores_all[0]["all_objective_scores"]
    auc_score = scores.pop("AUC")
    assert np.isnan(auc_score)
    assert not np.isnan(list(scores.values())).any()


def test_rankings(
    AutoMLTestEnv,
    X_y_binary,
    X_y_regression,
):
    X, y = X_y_binary
    model_families = ["random_forest"]
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_model_families=model_families,
        optimize_thresholds=False,
        max_iterations=3,
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.03}):
        automl.search()
    assert len(automl.full_rankings) == 3
    assert len(automl.rankings) == 2

    X, y = X_y_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        allowed_model_families=model_families,
        max_iterations=3,
        optimize_thresholds=False,
        n_jobs=1,
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={"R2": 0.03}):
        automl.search()
    assert len(automl.full_rankings) == 3
    assert len(automl.rankings) == 2


def test_automl_str_search(
    AutoMLTestEnv,
    X_y_binary,
):
    def _dummy_callback(pipeline, automl_obj):
        return None

    X, y = X_y_binary
    search_params = {
        "problem_type": "binary",
        "objective": "F1",
        "max_time": 100,
        "max_iterations": 5,
        "patience": 2,
        "tolerance": 0.5,
        "allowed_model_families": ["random_forest", "linear_model"],
        "data_splitter": StratifiedKFold(n_splits=5),
        "tuner_class": RandomSearchTuner,
        "start_iteration_callback": _dummy_callback,
        "add_result_callback": None,
        "additional_objectives": ["Precision", "AUC"],
        "n_jobs": 2,
        "optimize_thresholds": True,
    }

    param_str_reps = {
        "Objective": search_params["objective"],
        "Max Time": search_params["max_time"],
        "Max Iterations": search_params["max_iterations"],
        "Allowed Pipelines": [],
        "Patience": search_params["patience"],
        "Tolerance": search_params["tolerance"],
        "Data Splitting": "StratifiedKFold(n_splits=5, random_state=None, shuffle=False)",
        "Tuner": "RandomSearchTuner",
        "Start Iteration Callback": "_dummy_callback",
        "Add Result Callback": None,
        "Additional Objectives": search_params["additional_objectives"],
        "Random Seed": 0,
        "n_jobs": search_params["n_jobs"],
        "Optimize Thresholds": search_params["optimize_thresholds"],
    }

    automl = AutoMLSearch(X_train=X, y_train=y, **search_params)
    str_rep = str(automl)
    for param, value in param_str_reps.items():
        if isinstance(value, (tuple, list)):
            assert f"{param}" in str_rep
            for item in value:
                s = f"\t{str(item)}" if isinstance(value, list) else f"{item}"
                assert s in str_rep
        else:
            assert f"{param}: {str(value)}" in str_rep
    assert "Search Results" not in str_rep

    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    env.mock_fit.assert_called()
    env.mock_score.assert_called()
    env.mock_predict_proba.assert_called()
    env.mock_optimize_threshold.assert_called()

    str_rep = str(automl)
    assert "Search Results:" in str_rep
    assert automl.rankings.drop(["parameters"], axis="columns").to_string() in str_rep


def test_automl_str_no_param_search(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")

    param_str_reps = {
        "Objective": "Log Loss Binary",
        "Max Time": "None",
        "Max Iterations": "None",
        "Allowed Pipelines": [],
        "Patience": "None",
        "Tolerance": "0.0",
        "Data Splitting": "StratifiedKFold(n_splits=5, random_state=None, shuffle=False)",
        "Tuner": "SKOptTuner",
        "Additional Objectives": [
            "AUC",
            "Accuracy Binary",
            "Balanced Accuracy Binary",
            "F1",
            "Gini",
            "MCC Binary",
            "Precision",
        ],
        "Start Iteration Callback": "None",
        "Add Result Callback": "None",
        "Random Seed": 0,
        "n_jobs": "-1",
        "Optimize Thresholds": "False",
    }

    str_rep = str(automl)
    for param, value in param_str_reps.items():
        assert f"{param}" in str_rep
        if isinstance(value, list):
            value = "\n".join(["\t{}".format(item) for item in value])
            assert value in str_rep
    assert "Search Results" not in str_rep


def test_automl_feature_selection(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary

    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=2,
        start_iteration_callback=start_iteration_callback,
        allowed_component_graphs={
            "Name": [
                "RF Classifier Select From Model",
                "Logistic Regression Classifier",
            ]
        },
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1.0, "F1": 0.5}):
        automl.search()

    assert start_iteration_callback.call_count == 2
    proposed_parameters = start_iteration_callback.call_args_list[1][0][0].parameters
    assert proposed_parameters.keys() == {
        "RF Classifier Select From Model",
        "Logistic Regression Classifier",
    }
    assert (
        proposed_parameters["RF Classifier Select From Model"]["number_features"]
        == X.shape[1]
    )


@patch("evalml.tuners.random_search_tuner.RandomSearchTuner.is_search_space_exhausted")
def test_automl_tuner_exception(
    mock_is_search_space_exhausted, AutoMLTestEnv, X_y_binary
):
    X, y = X_y_binary
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    mock_is_search_space_exhausted.side_effect = NoParamsException(error_text)
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        tuner_class=RandomSearchTuner,
        max_iterations=10,
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with pytest.raises(NoParamsException, match=error_text):
        with env.test_context(score_return_value={"Log Loss Binary": 0.2}):
            automl.search()


@patch("evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch")
def test_automl_algorithm(
    mock_algo_next_batch,
    AutoMLTestEnv,
    X_y_binary,
):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=5)
    mock_algo_next_batch.side_effect = StopIteration("that's all, folks")
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1.0, "F1": 0.5}):
        automl.search()

    env.mock_fit.assert_called()
    env.mock_score.assert_called()
    assert mock_algo_next_batch.call_count == 1
    pipeline_results = automl.results.get("pipeline_results", {})
    assert len(pipeline_results) == 1
    assert pipeline_results[0].get("mean_cv_score") == 1.0


@patch("evalml.automl.automl_algorithm.IterativeAlgorithm.__init__")
def test_automl_allowed_component_graphs_algorithm(
    mock_algo_init,
    dummy_classifier_estimator_class,
    X_y_binary,
):
    mock_algo_init.side_effect = Exception("mock algo init")
    X, y = X_y_binary

    allowed_component_graphs = {
        "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class]
    }
    with pytest.raises(Exception, match="mock algo init"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            allowed_component_graphs=allowed_component_graphs,
            max_iterations=10,
        )
    assert mock_algo_init.call_count == 1
    _, kwargs = mock_algo_init.call_args
    assert kwargs["max_iterations"] == 10
    assert kwargs["allowed_pipelines"] == get_pipelines_from_component_graphs(
        allowed_component_graphs, "binary"
    )

    allowed_model_families = [ModelFamily.RANDOM_FOREST]
    with pytest.raises(Exception, match="mock algo init"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            allowed_model_families=allowed_model_families,
            max_iterations=1,
        )
    assert mock_algo_init.call_count == 2
    _, kwargs = mock_algo_init.call_args
    assert kwargs["max_iterations"] == 1
    for actual, expected in zip(
        kwargs["allowed_pipelines"],
        [
            make_pipeline(X, y, estimator, ProblemTypes.BINARY)
            for estimator in get_estimators(
                ProblemTypes.BINARY, model_families=allowed_model_families
            )
        ],
    ):
        assert actual.parameters == expected.parameters


@pytest.mark.parametrize("pickle_type", ["cloudpickle", "pickle", "invalid"])
def test_automl_serialization(pickle_type, X_y_binary, tmpdir):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), "automl.pkl")
    num_max_iterations = 5
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=num_max_iterations,
        n_jobs=1,
        verbose=True,
    )
    automl.search()

    if automl.search_iteration_plot:
        # Testing pickling of SearchIterationPlot object
        automl.search_iteration_plot = automl.plot.search_iteration_plot(
            interactive_plot=True
        )

    if pickle_type == "invalid":
        with pytest.raises(
            ValueError,
            match="`pickle_type` must be either 'pickle' or 'cloudpickle'. Received invalid",
        ):
            automl.save(path, pickle_type=pickle_type)
    else:
        automl.save(path, pickle_type=pickle_type)
        loaded_automl = automl.load(path)

        for i in range(num_max_iterations):
            assert (
                automl.get_pipeline(i).__class__
                == loaded_automl.get_pipeline(i).__class__
            )
            assert (
                automl.get_pipeline(i).parameters
                == loaded_automl.get_pipeline(i).parameters
            )

            for id_, pipeline_results in automl.results["pipeline_results"].items():
                loaded_ = loaded_automl.results["pipeline_results"][id_]
                for name in pipeline_results:
                    # Use np to check percent_better_than_baseline because of (possible) nans
                    if name == "percent_better_than_baseline_all_objectives":
                        for objective_name, value in pipeline_results[name].items():
                            np.testing.assert_almost_equal(
                                value, loaded_[name][objective_name]
                            )
                    elif name == "percent_better_than_baseline":
                        np.testing.assert_almost_equal(
                            pipeline_results[name], loaded_[name]
                        )
                    else:
                        assert pipeline_results[name] == loaded_[name]

        pd.testing.assert_frame_equal(automl.rankings, loaded_automl.rankings)


@patch("cloudpickle.dump")
def test_automl_serialization_protocol(mock_cloudpickle_dump, tmpdir, X_y_binary):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), "automl.pkl")
    automl = AutoMLSearch(
        X_train=X, y_train=y, problem_type="binary", max_iterations=5, n_jobs=1
    )

    automl.save(path)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert (
        mock_cloudpickle_dump.call_args_list[0][1]["protocol"]
        == cloudpickle.DEFAULT_PROTOCOL
    )

    mock_cloudpickle_dump.reset_mock()
    automl.save(path, pickle_protocol=42)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]["protocol"] == 42


def test_invalid_data_splitter(X_y_binary):
    X, y = X_y_binary
    data_splitter = pd.DataFrame()
    with pytest.raises(ValueError, match="Not a valid data splitter"):
        AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", data_splitter=data_splitter
        )


def test_large_dataset_binary(AutoMLTestEnv):
    X = pd.DataFrame({"col_0": [i for i in range(101000)]})
    y = pd.Series([i % 2 for i in range(101000)])

    fraud_objective = FraudCost(amount_col="col_0")

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective=fraud_objective,
        additional_objectives=["auc", "f1", "precision"],
        max_time=1,
        max_iterations=1,
        optimize_thresholds=True,
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.234}):
        automl.search()
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert automl.data_splitter.get_n_splits() == 1

    for pipeline_id in automl.results["search_order"]:
        assert len(automl.results["pipeline_results"][pipeline_id]["cv_data"]) == 1
        assert (
            automl.results["pipeline_results"][pipeline_id]["cv_data"][0][
                "mean_cv_score"
            ]
            == 1.234
        )
        assert (
            automl.results["pipeline_results"][pipeline_id]["mean_cv_score"]
            == automl.results["pipeline_results"][pipeline_id]["validation_score"]
        )


def test_large_dataset_multiclass(AutoMLTestEnv):
    X = pd.DataFrame({"col_0": [i for i in range(101000)]})
    y = pd.Series([i % 4 for i in range(101000)])

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        max_time=1,
        max_iterations=1,
        n_jobs=1,
    )
    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={automl.objective.name: 1.234}):
        automl.search()
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert automl.data_splitter.get_n_splits() == 1

    for pipeline_id in automl.results["search_order"]:
        assert len(automl.results["pipeline_results"][pipeline_id]["cv_data"]) == 1
        assert (
            automl.results["pipeline_results"][pipeline_id]["cv_data"][0][
                "mean_cv_score"
            ]
            == 1.234
        )
        assert (
            automl.results["pipeline_results"][pipeline_id]["mean_cv_score"]
            == automl.results["pipeline_results"][pipeline_id]["validation_score"]
        )


def test_large_dataset_regression(AutoMLTestEnv):
    X = pd.DataFrame({"col_0": [i for i in range(101000)]})
    y = pd.Series([i for i in range(101000)])

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        max_time=1,
        max_iterations=1,
        n_jobs=1,
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 1.234}):
        automl.search()
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert automl.data_splitter.get_n_splits() == 1

    for pipeline_id in automl.results["search_order"]:
        assert len(automl.results["pipeline_results"][pipeline_id]["cv_data"]) == 1
        assert (
            automl.results["pipeline_results"][pipeline_id]["cv_data"][0][
                "mean_cv_score"
            ]
            == 1.234
        )
        assert (
            automl.results["pipeline_results"][pipeline_id]["mean_cv_score"]
            == automl.results["pipeline_results"][pipeline_id]["validation_score"]
        )


def test_large_dataset_split_size(X_y_binary):
    X, y = X_y_binary

    def generate_fake_dataset(rows):
        X = pd.DataFrame({"col_0": [i for i in range(rows)]})
        y = pd.Series([i % 2 for i in range(rows)])
        return X, y

    fraud_objective = FraudCost(amount_col="col_0")

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective=fraud_objective,
        additional_objectives=["auc", "f1", "precision"],
        max_time=1,
        max_iterations=1,
        optimize_thresholds=True,
    )
    assert isinstance(automl.data_splitter, StratifiedKFold)

    under_max_rows = _LARGE_DATA_ROW_THRESHOLD - 1
    X, y = generate_fake_dataset(under_max_rows)
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective=fraud_objective,
        additional_objectives=["auc", "f1", "precision"],
        max_time=1,
        max_iterations=1,
        optimize_thresholds=True,
    )
    assert isinstance(automl.data_splitter, StratifiedKFold)

    automl.data_splitter = None
    over_max_rows = _LARGE_DATA_ROW_THRESHOLD + 1
    X, y = generate_fake_dataset(over_max_rows)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective=fraud_objective,
        additional_objectives=["auc", "f1", "precision"],
        max_time=1,
        max_iterations=1,
        optimize_thresholds=True,
    )
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    assert automl.data_splitter.test_size == (_LARGE_DATA_PERCENT_VALIDATION)


def test_data_splitter_shuffle():
    # this test checks that the default data split strategy should shuffle data. it creates a target which
    # increases monotonically from 0 to n-1.
    #
    # if shuffle is enabled, the baseline model, which predicts the mean of the training data, should accurately
    # predict the mean of the validation data, because the training split in each CV fold will contain a mix of
    # values from across the target range, thus yielding an R^2 of close to 0.
    #
    # if shuffle is disabled, the mean value learned on each CV fold's training data will be incredible inaccurate,
    # thus yielding an R^2 well below 0.

    n = 100000
    X = pd.DataFrame({"col_0": np.random.random(n)})
    y = pd.Series(np.arange(n), name="target")
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        max_time=1,
        max_iterations=1,
        n_jobs=1,
    )
    automl.search()
    assert automl.results["search_order"] == [0]
    assert len(automl.results["pipeline_results"][0]["cv_data"]) == 3
    for fold in range(3):
        np.testing.assert_almost_equal(
            automl.results["pipeline_results"][0]["cv_data"][fold]["mean_cv_score"],
            0.0,
            decimal=4,
        )
    np.testing.assert_almost_equal(
        automl.results["pipeline_results"][0]["mean_cv_score"], 0.0, decimal=4
    )
    np.testing.assert_almost_equal(
        automl.results["pipeline_results"][0]["validation_score"], 0.0, decimal=4
    )


def test_component_graph_with_incorrect_problem_type(
    dummy_classifier_estimator_class, X_y_binary
):
    X, y = X_y_binary
    # checks that not setting component graphs does not error out
    AutoMLSearch(X_train=X, y_train=y, problem_type="binary")

    with pytest.raises(ValueError, match="not valid for this component graph"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
            allowed_component_graphs={
                "Mock Binary Classification Pipeline": [
                    dummy_classifier_estimator_class
                ]
            },
        )


def test_main_objective_problem_type_mismatch(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", objective="R2")
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(
            X_train=X, y_train=y, problem_type="regression", objective="MCC Binary"
        )
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", objective="MCC Multiclass"
        )
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(X_train=X, y_train=y, problem_type="multiclass", objective="MSE")


def test_init_missing_data(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(
        ValueError,
        match=r"Must specify training data as a 2d array using the X_train argument",
    ):
        AutoMLSearch(y_train=y, problem_type="binary")

    with pytest.raises(
        ValueError,
        match=r"Must specify training data target values as a 1d vector using the y_train argument",
    ):
        AutoMLSearch(X_train=X, problem_type="binary")


def test_init_problem_type_error(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(
        ValueError,
        match=r"choose one of \(binary, multiclass, regression\) as problem_type",
    ):
        AutoMLSearch(X_train=X, y_train=y)

    with pytest.raises(KeyError, match=r"does not exist"):
        AutoMLSearch(X_train=X, y_train=y, problem_type="multi")


def test_init_objective(X_y_binary):
    X, y = X_y_binary
    defaults = {
        "multiclass": "Log Loss Multiclass",
        "binary": "Log Loss Binary",
        "regression": "R2",
    }
    for problem_type in defaults:
        error_automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type)
        assert error_automl.objective.name == defaults[problem_type]


@patch("evalml.automl.automl_search.AutoMLSearch.search")
def test_checks_at_search_time(mock_search, X_y_multi):
    X, y = X_y_multi

    error_text = "in search, problem_type mismatches label type."
    mock_search.side_effect = ValueError(error_text)

    error_automl = AutoMLSearch(
        X_train=X, y_train=y, problem_type="regression", objective="R2"
    )
    with pytest.raises(ValueError, match=error_text):
        error_automl.search()


def test_incompatible_additional_objectives(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match="is not compatible with a "):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="multiclass",
            additional_objectives=["Precision", "AUC"],
        )


def test_default_objective(X_y_binary):
    X, y = X_y_binary
    correct_matches = {
        ProblemTypes.MULTICLASS: "Log Loss Multiclass",
        ProblemTypes.BINARY: "Log Loss Binary",
        ProblemTypes.REGRESSION: "R2",
    }
    for problem_type in correct_matches:
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type)
        assert automl.objective.name == correct_matches[problem_type]

        automl = AutoMLSearch(X_train=X, y_train=y, problem_type=problem_type.name)
        assert automl.objective.name == correct_matches[problem_type]


def test_add_to_rankings(
    AutoMLTestEnv,
    dummy_binary_pipeline_class,
    X_y_binary,
):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1.0, "F1": 0.5}):
        automl.search()
    assert len(automl.rankings) == 1
    assert len(automl.full_rankings) == 1
    original_best_pipeline = automl.best_pipeline
    assert original_best_pipeline is not None

    with env.test_context(score_return_value={"Log Loss Binary": 0.1234}):
        test_pipeline = dummy_binary_pipeline_class(parameters={})
        automl.add_to_rankings(test_pipeline)
        assert automl.best_pipeline.name == test_pipeline.name
        assert automl.best_pipeline.parameters == test_pipeline.parameters
        assert automl.best_pipeline.component_graph == test_pipeline.component_graph

        assert len(automl.rankings) == 2
        assert len(automl.full_rankings) == 2
        assert 0.1234 in automl.rankings["mean_cv_score"].values

    with env.test_context(score_return_value={"Log Loss Binary": 0.5678}):
        test_pipeline_2 = dummy_binary_pipeline_class(
            parameters={"Mock Classifier": {"a": 1.234}}
        )
        automl.add_to_rankings(test_pipeline_2)
        assert automl.best_pipeline.name == test_pipeline.name
        assert automl.best_pipeline.parameters == test_pipeline.parameters
        assert automl.best_pipeline.component_graph == test_pipeline.component_graph
        assert len(automl.rankings) == 2
        assert len(automl.full_rankings) == 3
        assert 0.5678 not in automl.rankings["mean_cv_score"].values
        assert 0.5678 in automl.full_rankings["mean_cv_score"].values


def test_add_to_rankings_no_search(
    AutoMLTestEnv,
    dummy_binary_pipeline_class,
    X_y_binary,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.5234}):
        test_pipeline = dummy_binary_pipeline_class(parameters={})

        automl.add_to_rankings(test_pipeline)
        best_pipeline = automl.best_pipeline
        assert best_pipeline is not None
        assert isinstance(automl.data_splitter, StratifiedKFold)
        assert len(automl.rankings) == 1
        assert 0.5234 in automl.rankings["mean_cv_score"].values
        assert np.isnan(
            automl.results["pipeline_results"][0]["percent_better_than_baseline"]
        )
        assert all(
            np.isnan(res)
            for res in automl.results["pipeline_results"][0][
                "percent_better_than_baseline_all_objectives"
            ].values()
        )


def test_add_to_rankings_regression_large(
    AutoMLTestEnv,
    dummy_regression_pipeline_class,
    example_regression_graph,
):
    X = pd.DataFrame({"col_0": [i for i in range(101000)]})
    y = pd.Series([i for i in range(101000)])

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        allowed_component_graphs={"CG": example_regression_graph},
        problem_type="regression",
        max_time=1,
        max_iterations=1,
        n_jobs=1,
    )
    assert isinstance(automl.data_splitter, TrainingValidationSplit)
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 0.1234}):
        automl.add_to_rankings(dummy_regression_pipeline_class({}))
        assert isinstance(automl.data_splitter, TrainingValidationSplit)
        assert len(automl.rankings) == 1
        assert 0.1234 in automl.rankings["mean_cv_score"].values


def test_add_to_rankings_new_pipeline(dummy_regression_pipeline_class):
    X = pd.DataFrame({"col_0": [i for i in range(100)]})
    y = pd.Series([i for i in range(100)])

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        max_time=1,
        max_iterations=1,
        n_jobs=1,
    )
    test_pipeline = dummy_regression_pipeline_class(parameters={})
    automl.add_to_rankings(test_pipeline)


def test_add_to_rankings_regression(
    example_regression_graph,
    dummy_regression_pipeline_class,
    X_y_regression,
    AutoMLTestEnv,
):
    X, y = X_y_regression

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        allowed_component_graphs={"CG": example_regression_graph},
        problem_type="regression",
        max_time=1,
        max_iterations=1,
        n_jobs=1,
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 0.1234}):
        automl.add_to_rankings(dummy_regression_pipeline_class({}))

    assert isinstance(automl.data_splitter, KFold)
    assert len(automl.rankings) == 1
    assert 0.1234 in automl.rankings["mean_cv_score"].values


def test_add_to_rankings_duplicate(
    AutoMLTestEnv,
    dummy_binary_pipeline_class,
    X_y_binary,
):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.1234}):
        automl.search()
    best_pipeline = automl.best_pipeline
    test_pipeline = dummy_binary_pipeline_class(parameters={})
    assert automl.best_pipeline == best_pipeline
    automl.add_to_rankings(test_pipeline)

    test_pipeline_duplicate = dummy_binary_pipeline_class(parameters={})
    assert automl.add_to_rankings(test_pipeline_duplicate) is None


def test_add_to_rankings_trained(
    dummy_classifier_estimator_class,
    AutoMLTestEnv,
    dummy_binary_pipeline_class,
    X_y_binary,
):
    X, y = X_y_binary

    class CoolBinaryClassificationPipeline(dummy_binary_pipeline_class):
        custom_name = "Cool Binary Classification Pipeline"

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
        allowed_component_graphs={
            "Cool Binary Classification Pipeline": [dummy_classifier_estimator_class],
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1.0, "F1": 0.5}):
        automl.search()
    assert len(automl.rankings) == 1
    assert len(automl.full_rankings) == 1

    with env.test_context(score_return_value={"Log Loss Binary": 0.1234}):
        test_pipeline = dummy_binary_pipeline_class(parameters={})
        automl.add_to_rankings(test_pipeline)
        assert len(automl.rankings) == 2
        assert len(automl.full_rankings) == 2
        assert list(automl.rankings["mean_cv_score"].values).count(0.1234) == 1
        assert list(automl.full_rankings["mean_cv_score"].values).count(0.1234) == 1

    with env.test_context(
        score_return_value={"Log Loss Binary": 0.1234},
        mock_fit_return_value=CoolBinaryClassificationPipeline(parameters={}),
    ):
        test_pipeline_trained = CoolBinaryClassificationPipeline(parameters={}).fit(
            X, y
        )
        automl.add_to_rankings(test_pipeline_trained)
        assert len(automl.rankings) == 3
        assert len(automl.full_rankings) == 3
        assert list(automl.rankings["mean_cv_score"].values).count(0.1234) == 2
        assert list(automl.full_rankings["mean_cv_score"].values).count(0.1234) == 2


def test_no_search(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)

    df_columns = [
        "id",
        "pipeline_name",
        "search_order",
        "mean_cv_score",
        "standard_deviation_cv_score",
        "validation_score",
        "percent_better_than_baseline",
        "high_variance_cv",
        "parameters",
    ]
    assert (automl.rankings.columns == df_columns).all()
    assert (automl.full_rankings.columns == df_columns).all()

    with pytest.raises(PipelineNotFoundError):
        automl.best_pipeline

    with pytest.raises(PipelineNotFoundError):
        automl.get_pipeline(0)

    with pytest.raises(PipelineNotFoundError):
        automl.describe_pipeline(0)


def test_get_pipeline_invalid(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")
    with pytest.raises(
        PipelineNotFoundError, match="Pipeline not found in automl results"
    ):
        automl.get_pipeline(1000)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
        automl.search()
    assert automl.get_pipeline(0).name == "Mode Baseline Binary Classification Pipeline"
    automl._results["pipeline_results"][0].pop("pipeline_class")
    automl._pipelines_searched.pop(0)

    with pytest.raises(
        PipelineNotFoundError,
        match="Pipeline class or parameters not found in automl results",
    ):
        automl.get_pipeline(0)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=1,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
        automl.search()
    assert automl.get_pipeline(0).name == "Mode Baseline Binary Classification Pipeline"
    automl._results["pipeline_results"][0].pop("parameters")
    with pytest.raises(
        PipelineNotFoundError,
        match="Pipeline class or parameters not found in automl results",
    ):
        automl.get_pipeline(0)


def test_get_pipeline(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
        automl.search()
    automl.search()
    for _, ranking in automl.rankings.iterrows():
        pl = automl.get_pipeline(ranking.id)
        assert pl.parameters == ranking.parameters
        assert pl.name == ranking.pipeline_name
        assert not pl._is_fitted


@pytest.mark.parametrize("return_dict", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_describe_pipeline(return_dict, verbose, caplog, X_y_binary, AutoMLTestEnv):
    caplog.clear()
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=1,
        verbose=verbose,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
        automl.search()
    out = caplog.text

    assert ("Searching up to 1 pipelines. " in out) == verbose

    assert len(automl.results["pipeline_results"]) == 1
    caplog.clear()
    automl_dict = automl.describe_pipeline(0, return_dict=return_dict)
    out = caplog.text
    assert "Mode Baseline Binary Classification Pipeline" in out
    assert "Problem Type: binary" in out
    assert "Model Family: Baseline" in out
    assert "* strategy : mode" in out
    assert "Total training time (including CV): " in out
    assert "Log Loss Binary # Training # Validation" in out
    assert "0                      1.000         66           34" in out
    assert "1                      1.000         67           33" in out
    assert "2                      1.000         67           33" in out
    assert "mean                   1.000          -            -" in out
    assert "std                    0.000          -            -" in out
    assert "coef of var            0.000          -            -" in out

    if return_dict:
        assert automl_dict["id"] == 0
        assert (
            automl_dict["pipeline_name"]
            == "Mode Baseline Binary Classification Pipeline"
        )
        assert automl_dict["pipeline_summary"] == "Baseline Classifier"
        assert automl_dict["parameters"] == {
            "Baseline Classifier": {"strategy": "mode"}
        }
        assert automl_dict["mean_cv_score"] == 1.0
        assert not automl_dict["high_variance_cv"]
        assert isinstance(automl_dict["training_time"], float)
        assert automl_dict["cv_data"] == [
            {
                "all_objective_scores": OrderedDict(
                    [("Log Loss Binary", 1.0), ("# Training", 66), ("# Validation", 34)]
                ),
                "mean_cv_score": 1.0,
                "binary_classification_threshold": None,
            },
            {
                "all_objective_scores": OrderedDict(
                    [("Log Loss Binary", 1.0), ("# Training", 67), ("# Validation", 33)]
                ),
                "mean_cv_score": 1.0,
                "binary_classification_threshold": None,
            },
            {
                "all_objective_scores": OrderedDict(
                    [("Log Loss Binary", 1.0), ("# Training", 67), ("# Validation", 33)]
                ),
                "mean_cv_score": 1.0,
                "binary_classification_threshold": None,
            },
        ]
        assert automl_dict["percent_better_than_baseline_all_objectives"] == {
            "Log Loss Binary": 0
        }
        assert automl_dict["percent_better_than_baseline"] == 0
        assert automl_dict["validation_score"] == 1.0
    else:
        assert automl_dict is None


@pytest.mark.parametrize("return_dict", [True, False])
def test_describe_pipeline_with_ensembling(
    return_dict, X_y_binary, AutoMLTestEnv, caplog
):
    X, y = X_y_binary

    two_stacking_batches = 1 + 2 * (len(get_estimators(ProblemTypes.BINARY)) + 1)
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=two_stacking_batches,
        objective="Log Loss Binary",
        ensembling=True,
        optimize_thresholds=False,
        error_callback=raise_error_callback,
    )

    score_side_effect = [
        {"Log Loss Binary": score}
        for score in np.arange(
            0, -1 * automl.max_iterations * automl.data_splitter.get_n_splits(), -0.1
        )
    ]  # Dcreases with each call

    test_env = AutoMLTestEnv("binary")
    with test_env.test_context(mock_score_side_effect=score_side_effect):
        automl.search()
    pipeline_names = automl.rankings["pipeline_name"]
    assert pipeline_names.str.contains("Ensemble").any()

    ensemble_ids = [
        _get_first_stacked_classifier_no() - 1,
        _get_first_stacked_classifier_no(),
        len(automl.results["pipeline_results"]) - 2,
        len(automl.results["pipeline_results"]) - 1,
    ]

    num_sklearn_pl = 0

    for i, ensemble_id in enumerate(ensemble_ids):
        sklearn_pl = (
            True if "Sklearn" in automl.get_pipeline(ensemble_id).name else False
        )
        caplog.clear()
        automl_dict = automl.describe_pipeline(ensemble_id, return_dict=return_dict)
        out = caplog.text
        if sklearn_pl:
            assert "Sklearn Stacked Ensemble Classification Pipeline" in out
            assert "* final_estimator : None" in out
            num_sklearn_pl += 1
        else:
            assert "Stacked Ensemble Classification Pipeline" in out
            assert "* final_estimator : Elastic Net Classifier" in out
        assert "Problem Type: binary" in out
        assert "Model Family: Ensemble" in out
        assert "Total training time (including CV): " in out
        assert "Log Loss Binary # Training # Validation" in out
        assert "Input for ensembler are pipelines with IDs:" in out

        if return_dict:
            assert automl_dict["id"] == ensemble_id
            if sklearn_pl:
                assert (
                    automl_dict["pipeline_name"]
                    == "Sklearn Stacked Ensemble Classification Pipeline"
                )
                assert (
                    automl_dict["pipeline_summary"]
                    == "Sklearn Stacked Ensemble Classifier"
                )
            else:
                assert (
                    automl_dict["pipeline_name"]
                    == "Stacked Ensemble Classification Pipeline"
                )
                assert "Stacked Ensemble Classifier" in automl_dict["pipeline_summary"]
            assert isinstance(automl_dict["mean_cv_score"], float)
            assert not automl_dict["high_variance_cv"]
            assert isinstance(automl_dict["training_time"], float)
            assert isinstance(
                automl_dict["percent_better_than_baseline_all_objectives"], dict
            )
            assert isinstance(automl_dict["percent_better_than_baseline"], float)
            assert isinstance(automl_dict["validation_score"], float)
            assert len(automl_dict["input_pipeline_ids"]) == len(
                allowed_model_families("binary")
            )
            if i < 2:
                assert all(
                    input_id < ensemble_id
                    for input_id in automl_dict["input_pipeline_ids"]
                )
            else:
                assert all(
                    input_id < ensemble_id
                    for input_id in automl_dict["input_pipeline_ids"]
                )
                assert all(
                    input_id > ensemble_ids[0]
                    for input_id in automl_dict["input_pipeline_ids"]
                )
        else:
            assert automl_dict is None
    assert num_sklearn_pl == 2


def test_results_getter(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=1,
    )
    env = AutoMLTestEnv("binary")

    assert automl.results == {"pipeline_results": {}, "search_order": []}
    with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
        automl.search()

    assert automl.results["pipeline_results"][0]["mean_cv_score"] == 1.0

    with pytest.raises(AttributeError, match="set attribute"):
        automl.results = 2.0

    automl.results["pipeline_results"][0]["mean_cv_score"] = 2.0
    assert automl.results["pipeline_results"][0]["mean_cv_score"] == 1.0


@pytest.mark.parametrize("data_type", ["li", "np", "pd", "ww"])
@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@pytest.mark.parametrize(
    "target_type",
    [
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "bool",
        "category",
        "object",
    ],
)
def test_targets_pandas_data_types_classification(
    breast_cancer_local, wine_local, data_type, automl_type, target_type, make_data_type
):
    if data_type == "np" and target_type in ["Int64", "boolean"]:
        pytest.skip(
            "Skipping test where data type is numpy and target type is nullable dtype"
        )

    if automl_type == ProblemTypes.BINARY:
        X, y = breast_cancer_local
        if "bool" in target_type:
            y = y.map({"malignant": False, "benign": True})
    elif automl_type == ProblemTypes.MULTICLASS:
        if "bool" in target_type:
            pytest.skip(
                "Skipping test where problem type is multiclass but target type is boolean"
            )
        X, y = wine_local
    unique_vals = y.unique()
    # Update target types as necessary
    if target_type in ["category", "object"]:
        if target_type == "category":
            y = pd.Series(pd.Categorical(y))
    elif "int" in target_type.lower():
        y = y.map({unique_vals[i]: int(i) for i in range(len(unique_vals))})
    elif "float" in target_type.lower():
        y = y.map({unique_vals[i]: float(i) for i in range(len(unique_vals))})

    y = y.astype(target_type)
    if data_type != "pd":
        X = make_data_type(data_type, X)
        y = make_data_type(data_type, y)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        max_iterations=3,
        n_jobs=1,
    )
    automl.search()
    for pipeline_id, pipeline_result in automl.results["pipeline_results"].items():
        cv_data = pipeline_result["cv_data"]
        for fold in cv_data:
            all_objective_scores = fold["all_objective_scores"]
            for score in all_objective_scores.values():
                assert score is not None

    assert len(automl.full_rankings) == 3
    assert not automl.full_rankings["mean_cv_score"].isnull().values.any()


class KeyboardInterruptOnKthPipeline:
    """Helps us time when the test will send a KeyboardInterrupt Exception to search."""

    def __init__(self, k, starting_index):
        self.n_calls = starting_index
        self.k = k

    def __call__(self):
        """Raises KeyboardInterrupt on the kth call.
        Arguments are ignored but included to meet the call back API.
        """
        if self.n_calls == self.k:
            self.n_calls += 1
            raise KeyboardInterrupt
        else:
            self.n_calls += 1
            return True


# These are used to mock return values to the builtin "input" function.
interrupt = ["y"]
interrupt_after_bad_message = ["No.", "Yes!", "y"]
dont_interrupt = ["n"]
dont_interrupt_after_bad_message = ["Yes", "yes.", "n"]


@pytest.mark.parametrize(
    "when_to_interrupt,user_input,number_results",
    [(1, interrupt, 0), (1, interrupt_after_bad_message, 0)],
)
@patch("builtins.input")
@patch("evalml.automl.engine.sequential_engine.SequentialComputation.get_result")
def test_catch_keyboard_interrupt_baseline(
    mock_future_get_result,
    mock_input,
    when_to_interrupt,
    user_input,
    number_results,
    X_y_binary,
    AutoMLTestEnv,
):
    X, y = X_y_binary

    mock_input.side_effect = user_input
    mock_future_get_result.side_effect = KeyboardInterruptOnKthPipeline(
        k=when_to_interrupt, starting_index=1
    )
    automl = AutoMLSearch(
        X_train=X, y_train=y, problem_type="binary", max_iterations=5, objective="f1"
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"F1": 1.0}):
        automl.search()
    assert len(automl._results["pipeline_results"]) == number_results
    if number_results == 0:
        with pytest.raises(PipelineNotFoundError):
            _ = automl.best_pipeline


@pytest.mark.parametrize(
    "when_to_interrupt,user_input,number_results",
    [
        (1, dont_interrupt, 5),
        (1, dont_interrupt_after_bad_message, 5),
        (2, interrupt, 1),
        (2, interrupt_after_bad_message, 1),
        (2, dont_interrupt, 5),
        (2, dont_interrupt_after_bad_message, 5),
        (3, interrupt, 2),
        (3, interrupt_after_bad_message, 2),
        (3, dont_interrupt, 5),
        (3, dont_interrupt_after_bad_message, 5),
        (5, interrupt, 4),
        (5, interrupt_after_bad_message, 4),
        (5, dont_interrupt, 5),
        (5, dont_interrupt_after_bad_message, 5),
    ],
)
@patch("builtins.input")
@patch("evalml.automl.engine.sequential_engine.SequentialComputation.done")
def test_catch_keyboard_interrupt(
    mock_future_get_result,
    mock_input,
    when_to_interrupt,
    user_input,
    number_results,
    X_y_binary,
    AutoMLTestEnv,
):
    X, y = X_y_binary

    mock_input.side_effect = user_input
    mock_future_get_result.side_effect = KeyboardInterruptOnKthPipeline(
        k=when_to_interrupt, starting_index=2
    )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=5,
        objective="f1",
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1}):
        automl.search()
    assert len(automl._results["pipeline_results"]) == number_results


@patch("builtins.input", return_value="Y")
@patch(
    "evalml.automl.engine.sequential_engine.SequentialComputation.done",
    side_effect=KeyboardInterruptOnKthPipeline(k=4, starting_index=2),
)
@patch("evalml.automl.engine.sequential_engine.SequentialComputation.cancel")
def test_jobs_cancelled_when_keyboard_interrupt(
    mock_cancel,
    mock_done,
    mock_input,
    X_y_binary,
    AutoMLTestEnv,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=6,
        objective="f1",
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"F1": 1}):
        automl.search()
    assert len(automl._results["pipeline_results"]) == 3

    # Since we trigger KeyBoardInterrupt the 4th time we call done, we've successfully evaluated the baseline plus 2
    # pipelines in the first batch. Since there are len(automl.allowed_pipelines) pipelines in the first batch,
    # we should cancel len(automl.allowed_pipelines) - 2 computations
    assert mock_cancel.call_count == len(automl.allowed_pipelines) - 3 + 1


def make_mock_rankings(scores):
    df = pd.DataFrame(
        {
            "id": range(len(scores)),
            "mean_cv_score": scores,
            "pipeline_name": [f"Mock name {i}" for i in range(len(scores))],
        }
    )
    return df


@patch("evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch")
@patch("evalml.automl.AutoMLSearch.full_rankings", new_callable=PropertyMock)
@patch("evalml.automl.AutoMLSearch.rankings", new_callable=PropertyMock)
def test_pipelines_in_batch_return_nan(
    mock_rankings,
    mock_full_rankings,
    mock_next_batch,
    X_y_binary,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline_class,
    AutoMLTestEnv,
):
    X, y = X_y_binary
    mock_rankings.side_effect = [
        make_mock_rankings([0, 0, 0]),  # first batch
        make_mock_rankings([0, 0, 0, 0, np.nan]),  # second batch
        make_mock_rankings([0, 0, 0, 0, np.nan, np.nan, np.nan]),
    ]  # third batch, should raise error
    mock_full_rankings.side_effect = [
        make_mock_rankings([0, 0, 0]),  # first batch
        make_mock_rankings([0, 0, 0, 0, np.nan]),  # second batch
        make_mock_rankings([0, 0, 0, 0, np.nan, np.nan, np.nan]),
    ]  # third batch, should raise error
    mock_next_batch.side_effect = [
        [
            dummy_binary_pipeline_class(parameters={}),
            dummy_binary_pipeline_class(parameters={}),
        ]
        for i in range(3)
    ]
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=3,
        allowed_component_graphs={"Name": [dummy_classifier_estimator_class]},
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with pytest.raises(
        AutoMLSearchException,
        match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective",
    ):
        with env.test_context(score_return_value={"Log Loss Binary": None}):
            automl.search()


@patch("evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch")
@patch("evalml.automl.AutoMLSearch.full_rankings", new_callable=PropertyMock)
@patch("evalml.automl.AutoMLSearch.rankings", new_callable=PropertyMock)
def test_pipelines_in_batch_return_none(
    mock_rankings,
    mock_full_rankings,
    mock_next_batch,
    X_y_binary,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline_class,
    AutoMLTestEnv,
):
    X, y = X_y_binary
    mock_rankings.side_effect = [
        make_mock_rankings([0, 0, 0]),  # first batch
        make_mock_rankings([0, 0, 0, 0, None]),  # second batch
        make_mock_rankings([0, 0, 0, 0, None, None, None]),
    ]  # third batch, should raise error
    mock_full_rankings.side_effect = [
        make_mock_rankings([0, 0, 0]),  # first batch
        make_mock_rankings([0, 0, 0, 0, None]),  # second batch
        make_mock_rankings([0, 0, 0, 0, None, None, None]),
    ]  # third batch, should raise error
    mock_next_batch.side_effect = [
        [
            dummy_binary_pipeline_class(parameters={}),
            dummy_binary_pipeline_class(parameters={}),
        ]
        for i in range(3)
    ]
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=3,
        allowed_component_graphs={"Name": [dummy_classifier_estimator_class]},
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with pytest.raises(
        AutoMLSearchException,
        match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective",
    ):
        with env.test_context(score_return_value={"Log Loss Binary": None}):
            automl.search()


@patch("evalml.automl.engine.engine_base.split_data")
def test_error_during_train_test_split(mock_split_data, X_y_binary, AutoMLTestEnv):
    X, y = X_y_binary
    # this method is called during pipeline eval for binary classification and will cause scores to be set to nan
    mock_split_data.side_effect = RuntimeError()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective="Accuracy Binary",
        max_iterations=2,
        optimize_thresholds=False,
        train_best_pipeline=False,
    )
    env = AutoMLTestEnv("binary")
    with pytest.raises(
        AutoMLSearchException,
        match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective",
    ):
        with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
            automl.search()
    for pipeline in automl.results["pipeline_results"].values():
        assert np.isnan(pipeline["mean_cv_score"])


all_objectives = (
    get_core_objectives("binary")
    + get_core_objectives("multiclass")
    + get_core_objectives("regression")
)


class CustomClassificationObjective(BinaryClassificationObjective):
    """Accuracy score for binary and multiclass classification."""

    name = "Classification Accuracy"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False
    expected_range = [0, 1]
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def objective_function(self, y_true, y_predicted, X=None):
        """Not implementing since mocked in our tests."""


class CustomRegressionObjective(RegressionObjective):
    """Accuracy score for binary and multiclass classification."""

    name = "Custom Regression Objective"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False
    expected_range = [0, 1]
    problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def objective_function(self, y_true, y_predicted, X=None):
        """Not implementing since mocked in our tests."""


@pytest.mark.parametrize(
    "objective,pipeline_scores,baseline_score,problem_type_value",
    product(
        all_objectives + [CostBenefitMatrix, CustomClassificationObjective()],
        [(0.3, 0.4), (np.nan, 0.4), (0.3, np.nan), (np.nan, np.nan)],
        [0.1, np.nan],
        [
            ProblemTypes.BINARY,
            ProblemTypes.MULTICLASS,
            ProblemTypes.REGRESSION,
            ProblemTypes.TIME_SERIES_REGRESSION,
        ],
    ),
)
@patch("evalml.tuners.skopt_tuner.Optimizer.tell")
def test_percent_better_than_baseline_in_rankings(
    mock_tell,
    objective,
    pipeline_scores,
    baseline_score,
    problem_type_value,
    dummy_binary_pipeline_class,
    dummy_multiclass_pipeline_class,
    dummy_regression_pipeline_class,
    dummy_time_series_regression_pipeline_class,
    X_y_binary,
):
    if not objective.is_defined_for_problem_type(problem_type_value):
        pytest.skip("Skipping because objective is not defined for problem type")

    # Ok to only use binary labels since score and fit methods are mocked
    X, y = X_y_binary

    pipeline_class = {
        ProblemTypes.BINARY: dummy_binary_pipeline_class,
        ProblemTypes.MULTICLASS: dummy_multiclass_pipeline_class,
        ProblemTypes.REGRESSION: dummy_regression_pipeline_class,
        ProblemTypes.TIME_SERIES_REGRESSION: dummy_time_series_regression_pipeline_class,
    }[problem_type_value]
    baseline_pipeline_class = {
        ProblemTypes.BINARY: "evalml.pipelines.BinaryClassificationPipeline",
        ProblemTypes.MULTICLASS: "evalml.pipelines.MulticlassClassificationPipeline",
        ProblemTypes.REGRESSION: "evalml.pipelines.RegressionPipeline",
        ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesRegressionPipeline",
    }[problem_type_value]

    class DummyPipeline(pipeline_class):
        problem_type = problem_type_value

        def __init__(self, parameters, random_seed=0):
            super().__init__(parameters=parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

        def fit(self, *args, **kwargs):
            """Mocking fit"""

    class Pipeline1(DummyPipeline):
        custom_name = "Pipeline1"

    class Pipeline2(DummyPipeline):
        custom_name = "Pipeline2"

    mock_score_1 = MagicMock(return_value={objective.name: pipeline_scores[0]})
    mock_score_2 = MagicMock(return_value={objective.name: pipeline_scores[1]})
    Pipeline1.score = mock_score_1
    Pipeline2.score = mock_score_2

    pipeline_parameters = (
        {
            "pipeline": {
                "date_index": None,
                "gap": 0,
                "max_delay": 0,
                "forecast_horizon": 2,
            }
        }
        if problem_type_value == ProblemTypes.TIME_SERIES_REGRESSION
        else {}
    )
    allowed_pipelines = [Pipeline1(pipeline_parameters), Pipeline2(pipeline_parameters)]

    if objective.name.lower() == "cost benefit matrix":
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type=problem_type_value,
            max_iterations=3,
            objective=objective(0, 0, 0, 0),
            additional_objectives=[],
            optimize_thresholds=False,
            n_jobs=1,
        )
    elif problem_type_value == ProblemTypes.TIME_SERIES_REGRESSION:
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type=problem_type_value,
            max_iterations=3,
            objective=objective,
            additional_objectives=[],
            problem_configuration={
                "date_index": None,
                "gap": 0,
                "max_delay": 0,
                "forecast_horizon": 2,
            },
            train_best_pipeline=False,
            n_jobs=1,
        )
    else:
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type=problem_type_value,
            max_iterations=3,
            objective=objective,
            additional_objectives=[],
            optimize_thresholds=False,
            n_jobs=1,
        )
    automl._automl_algorithm = IterativeAlgorithm(
        max_iterations=2,
        allowed_pipelines=allowed_pipelines,
        tuner_class=SKOptTuner,
        random_seed=0,
        n_jobs=1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        pipeline_params=pipeline_parameters,
        custom_hyperparameters=None,
    )
    automl._SLEEP_TIME = 0.000001
    with patch(
        baseline_pipeline_class + ".score",
        return_value={objective.name: baseline_score},
    ):
        if np.isnan(pipeline_scores).all():
            with pytest.raises(
                AutoMLSearchException,
                match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective",
            ):
                automl.search()
        else:
            automl.search()
        scores = dict(
            zip(
                automl.rankings.pipeline_name,
                automl.rankings.percent_better_than_baseline,
            )
        )
        baseline_name = next(
            name
            for name in automl.rankings.pipeline_name
            if name not in {"Pipeline1", "Pipeline2"}
        )
        answers = {
            "Pipeline1": round(
                objective.calculate_percent_difference(
                    pipeline_scores[0], baseline_score
                ),
                2,
            ),
            "Pipeline2": round(
                objective.calculate_percent_difference(
                    pipeline_scores[1], baseline_score
                ),
                2,
            ),
            baseline_name: round(
                objective.calculate_percent_difference(baseline_score, baseline_score),
                2,
            ),
        }
        for name in answers:
            np.testing.assert_almost_equal(scores[name], answers[name], decimal=3)


@pytest.mark.parametrize("custom_additional_objective", [True, False])
@pytest.mark.parametrize(
    "problem_type", ["binary", "multiclass", "regression", "time series regression"]
)
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
@patch("evalml.pipelines.MulticlassClassificationPipeline.fit")
@patch("evalml.pipelines.RegressionPipeline.fit")
@patch("evalml.pipelines.TimeSeriesRegressionPipeline.fit")
@patch("evalml.tuners.skopt_tuner.Optimizer.tell")
def test_percent_better_than_baseline_computed_for_all_objectives(
    mock_tell,
    mock_time_series_baseline_regression_fit,
    mock_regression_fit,
    mock_multiclass_fit,
    mock_binary_fit,
    problem_type,
    custom_additional_objective,
    dummy_binary_pipeline_class,
    dummy_multiclass_pipeline_class,
    dummy_regression_pipeline_class,
    dummy_time_series_regression_pipeline_class,
    X_y_binary,
):
    X, y = X_y_binary

    problem_type_enum = handle_problem_types(problem_type)

    pipeline_class = {
        "binary": dummy_binary_pipeline_class,
        "multiclass": dummy_multiclass_pipeline_class,
        "regression": dummy_regression_pipeline_class,
        "time series regression": dummy_time_series_regression_pipeline_class,
    }[problem_type]
    baseline_pipeline_class = {
        ProblemTypes.BINARY: "evalml.pipelines.BinaryClassificationPipeline",
        ProblemTypes.MULTICLASS: "evalml.pipelines.MulticlassClassificationPipeline",
        ProblemTypes.REGRESSION: "evalml.pipelines.RegressionPipeline",
        ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesRegressionPipeline",
    }[problem_type_enum]

    class DummyPipeline(pipeline_class):
        name = "Dummy 1"
        problem_type = problem_type_enum

        def __init__(self, parameters, random_seed=0):
            super().__init__(parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

        def fit(self, *args, **kwargs):
            """Mocking fit"""

    additional_objectives = None
    if custom_additional_objective:
        if CustomClassificationObjective.is_defined_for_problem_type(problem_type_enum):
            additional_objectives = [CustomClassificationObjective()]
        else:
            additional_objectives = [
                CustomRegressionObjective(),
                "Root Mean Squared Error",
            ]

    core_objectives = get_core_objectives(problem_type)
    if additional_objectives:
        core_objectives = [
            get_default_primary_search_objective(problem_type_enum)
        ] + additional_objectives
    mock_scores = {get_objective(obj).name: i for i, obj in enumerate(core_objectives)}
    mock_baseline_scores = {
        get_objective(obj).name: i + 1 for i, obj in enumerate(core_objectives)
    }
    answer = {}
    baseline_percent_difference = {}
    for obj in core_objectives:
        obj_class = get_objective(obj)
        answer[obj_class.name] = obj_class.calculate_percent_difference(
            mock_scores[obj_class.name], mock_baseline_scores[obj_class.name]
        )
        baseline_percent_difference[obj_class.name] = 0

    mock_score_1 = MagicMock(return_value=mock_scores)
    DummyPipeline.score = mock_score_1
    parameters = {}
    if problem_type_enum == ProblemTypes.TIME_SERIES_REGRESSION:
        parameters = {
            "pipeline": {
                "date_index": None,
                "gap": 6,
                "max_delay": 3,
                "forecast_horizon": 3,
            }
        }
    # specifying problem_configuration for all problem types for conciseness
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        max_iterations=2,
        objective="auto",
        problem_configuration={
            "date_index": None,
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 3,
        },
        optimize_thresholds=False,
        additional_objectives=additional_objectives,
    )
    automl._automl_algorithm = IterativeAlgorithm(
        max_iterations=2,
        allowed_pipelines=[DummyPipeline(parameters)],
        tuner_class=SKOptTuner,
        random_seed=0,
        n_jobs=-1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        pipeline_params={
            "pipeline": {
                "date_index": None,
                "gap": 1,
                "max_delay": 1,
                "forecast_horizon": 2,
            }
        },
        custom_hyperparameters=None,
    )
    automl._SLEEP_TIME = 0.00001
    with patch(baseline_pipeline_class + ".score", return_value=mock_baseline_scores):
        automl.search()
        assert (
            len(automl.results["pipeline_results"]) == 2
        ), "This tests assumes only one non-baseline pipeline was run!"
        pipeline_results = automl.results["pipeline_results"][1]
        baseline_results = automl.results["pipeline_results"][0]
        assert pipeline_results["percent_better_than_baseline_all_objectives"] == answer
        assert (
            pipeline_results["percent_better_than_baseline"]
            == pipeline_results["percent_better_than_baseline_all_objectives"][
                automl.objective.name
            ]
        )
        # Check that baseline is 0% better than baseline
        assert (
            baseline_results["percent_better_than_baseline_all_objectives"]
            == baseline_percent_difference
        )


def test_time_series_regression_with_parameters(ts_data):
    X, y = ts_data
    X.index.name = "Date"
    problem_configuration = {
        "date_index": "Date",
        "gap": 1,
        "max_delay": 0,
        "forecast_horizon": 2,
    }
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="time series regression",
        allowed_component_graphs={"Name_0": ["Imputer", "Linear Regressor"]},
        objective="auto",
        problem_configuration=problem_configuration,
        max_batches=3,
    )
    assert automl.allowed_pipelines[0].parameters["pipeline"] == problem_configuration


@pytest.mark.parametrize("graph_type", ["dict", "cg"])
def test_automl_accepts_component_graphs(graph_type, X_y_binary):
    X, y = X_y_binary
    if graph_type == "dict":
        component_graph = {
            "imputer": ["Imputer", "X", "y"],
            "ohe": ["One Hot Encoder", "imputer.x", "y"],
            "estimator_1": ["Random Forest Classifier", "ohe.x", "y"],
            "estimator_2": ["Decision Tree Classifier", "ohe.x", "y"],
            "final": [
                "Logistic Regression Classifier",
                "estimator_1.x",
                "estimator_2.x",
                "y",
            ],
        }
        component_graph_obj = ComponentGraph(component_graph)
    else:
        component_dict = {
            "imputer": ["Imputer", "X", "y"],
            "ohe": ["One Hot Encoder", "imputer.x", "y"],
            "estimator_1": ["Random Forest Classifier", "ohe.x", "y"],
            "estimator_2": ["Decision Tree Classifier", "ohe.x", "y"],
            "final": [
                "Logistic Regression Classifier",
                "estimator_1.x",
                "estimator_2.x",
                "y",
            ],
        }
        component_graph = ComponentGraph(component_dict)
        component_graph_obj = component_graph
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs={"Dummy_Name": component_graph},
        objective="auto",
        max_batches=3,
    )
    for pipeline_ in automl.allowed_pipelines:
        assert isinstance(pipeline_, BinaryClassificationPipeline)
        assert pipeline_.component_graph == component_graph_obj


@pytest.mark.parametrize("fold_scores", [[2, 4, 6], [np.nan, 4, 6]])
def test_percent_better_than_baseline_scores_different_folds(
    fold_scores,
    dummy_binary_pipeline_class,
    X_y_binary,
    AutoMLTestEnv,
):
    # Test that percent-better-than-baseline is correctly computed when scores differ across folds
    X, y = X_y_binary

    class DummyPipeline(dummy_binary_pipeline_class):
        name = "Dummy 1"
        problem_type = ProblemTypes.BINARY

        def __init__(self, parameters, random_seed=0):
            super().__init__(parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    mock_score = MagicMock(
        side_effect=[{"Log Loss Binary": 1, "F1": val} for val in fold_scores]
    )
    DummyPipeline.score = mock_score
    f1 = get_objective("f1")()

    if np.isnan(fold_scores[0]):
        answer = np.nan
    else:
        answer = f1.calculate_percent_difference(4, 1)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=2,
        objective="log loss binary",
        optimize_thresholds=False,
        additional_objectives=["f1"],
    )
    automl._automl_algorithm = IterativeAlgorithm(
        max_iterations=2,
        allowed_pipelines=[DummyPipeline({})],
        tuner_class=SKOptTuner,
        random_seed=0,
        n_jobs=-1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        pipeline_params={},
        custom_hyperparameters=None,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1, "F1": 1}):
        automl.search()
    assert (
        len(automl.results["pipeline_results"]) == 2
    ), "This tests assumes only one non-baseline pipeline was run!"
    pipeline_results = automl.results["pipeline_results"][1]
    np.testing.assert_equal(
        pipeline_results["percent_better_than_baseline_all_objectives"]["F1"], answer
    )


def _get_first_stacked_classifier_no(model_families=None):
    """Gets the number of iterations necessary before the stacked ensemble will be used."""
    num_classifiers = len(
        get_estimators(ProblemTypes.BINARY, model_families=model_families)
    )
    # Baseline + first batch + each pipeline iteration (5 is current default pipelines_per_batch) + 1
    return 1 + num_classifiers + num_classifiers * 5 + 1


@pytest.mark.parametrize(
    "max_iterations",
    [
        None,
        1,
        8,
        10,
        _get_first_stacked_classifier_no(),
        _get_first_stacked_classifier_no() + 2,
    ],
)
@pytest.mark.parametrize("use_ensembling", [True, False])
def test_max_iteration_works_with_stacked_ensemble(
    max_iterations, use_ensembling, AutoMLTestEnv, X_y_binary, caplog
):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations,
        objective="Log Loss Binary",
        optimize_thresholds=False,
        ensembling=use_ensembling,
        verbose=True,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.8}):
        automl.search()
    # every nth batch a stacked ensemble will be trained
    if max_iterations is None:
        max_iterations = 5  # Default value for max_iterations

    pipeline_names = automl.rankings["pipeline_name"]
    if max_iterations < _get_first_stacked_classifier_no():
        assert not pipeline_names.str.contains("Ensemble").any()
    elif use_ensembling:
        assert pipeline_names.str.contains("Ensemble").any()
        assert (
            f"Ensembling will run at the {_get_first_stacked_classifier_no()} iteration"
            in caplog.text
        )

    else:
        assert not pipeline_names.str.contains("Ensemble").any()


@pytest.mark.parametrize("max_batches", [None, 1, 5, 8, 9, 10, 12, 20])
@pytest.mark.parametrize("use_ensembling", [True, False])
@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.REGRESSION])
def test_max_batches_works(
    max_batches,
    use_ensembling,
    problem_type,
    AutoMLTestEnv,
    X_y_binary,
    X_y_regression,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        max_iterations=None,
        max_batches=max_batches,
        optimize_thresholds=False,
        ensembling=use_ensembling,
    )
    env = AutoMLTestEnv(problem_type)
    with env.test_context(score_return_value={automl.objective.name: 0.3}):
        automl.search()
    # every nth batch a stacked ensemble will be trained
    ensemble_nth_batch = len(automl.allowed_pipelines) + 1

    if max_batches is None:
        n_results = len(automl.allowed_pipelines) + 1
        max_batches = 1
        # _automl_algorithm will include all allowed_pipelines in the first batch even
        # if they are not searched over. That is why n_automl_pipelines does not equal
        # n_results when max_iterations and max_batches are None
        n_automl_pipelines = 1 + len(automl.allowed_pipelines)
        num_ensemble_batches = 0
    else:
        # automl algorithm does not know about the additional stacked ensemble pipelines
        num_ensemble_batches = (
            (max_batches - 1) // ensemble_nth_batch if use_ensembling else 0
        )
        # So that the test does not break when new estimator classes are added
        n_results = (
            1
            + len(automl.allowed_pipelines)
            + (5 * (max_batches - 1 - num_ensemble_batches))
            + num_ensemble_batches * 2
        )
        n_automl_pipelines = n_results
    assert automl._automl_algorithm.batch_number == max_batches
    assert automl._automl_algorithm.pipeline_number + 1 == n_automl_pipelines
    assert len(automl.results["pipeline_results"]) == n_results
    if num_ensemble_batches == 0:
        assert automl.rankings.shape[0] == min(
            1 + len(automl.allowed_pipelines), n_results
        )  # add one for baseline
    else:
        assert automl.rankings.shape[0] == min(
            3 + len(automl.allowed_pipelines), n_results
        )  # add two for baseline and two for stacked ensemble
    assert automl.full_rankings.shape[0] == n_results


def test_early_stopping_negative(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match="patience value must be a positive integer."):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            objective="AUC",
            max_iterations=5,
            allowed_model_families=["linear_model"],
            patience=-1,
            random_seed=0,
        )
    with pytest.raises(ValueError, match="tolerance value must be"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            objective="AUC",
            max_iterations=5,
            allowed_model_families=["linear_model"],
            patience=1,
            tolerance=1.5,
            random_seed=0,
        )


@pytest.mark.parametrize("verbose", [True, False])
def test_early_stopping(
    verbose, caplog, logistic_regression_binary_pipeline_class, X_y_binary
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective="AUC",
        max_iterations=5,
        allowed_model_families=["linear_model"],
        patience=2,
        tolerance=0.05,
        random_seed=0,
        n_jobs=1,
        verbose=verbose,
    )
    mock_results = {"search_order": [0, 1, 2, 3], "pipeline_results": {}}

    scores = [
        0.84,
        0.95,
        0.84,
        0.96,
    ]  # 0.96 is only 1% greater so it doesn't trigger patience due to tolerance
    for id in mock_results["search_order"]:
        mock_results["pipeline_results"][id] = {}
        mock_results["pipeline_results"][id]["mean_cv_score"] = scores[id]
        mock_results["pipeline_results"][id][
            "pipeline_class"
        ] = logistic_regression_binary_pipeline_class
    automl._results = mock_results

    assert not automl._should_continue()
    out = caplog.text
    assert (
        "2 iterations without improvement. Stopping search early." in out
    ) == verbose


def test_automl_one_allowed_component_graph_ensembling_disabled(
    AutoMLTestEnv,
    X_y_binary,
    caplog,
):
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.RANDOM_FOREST]) + 1
    # Checks that when len(allowed_component_graphs) == 1, ensembling is not run, even if set to True
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations,
        optimize_thresholds=False,
        allowed_model_families=[ModelFamily.RANDOM_FOREST],
        ensembling=True,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    assert (
        "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run."
        in caplog.text
    )

    pipeline_names = automl.rankings["pipeline_name"]
    assert not pipeline_names.str.contains("Ensemble").any()

    caplog.clear()
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.LINEAR_MODEL]) + 1
    allowed_component_graph = {
        "Logistic Regression Binary Pipeline": [
            "Imputer",
            "One Hot Encoder",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ]
    }
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations,
        allowed_component_graphs=allowed_component_graph,
        optimize_thresholds=False,
        ensembling=True,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    pipeline_names = automl.rankings["pipeline_name"]
    assert not pipeline_names.str.contains("Ensemble").any()
    assert (
        "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run."
        in caplog.text
    )
    # Check that ensembling runs when len(allowed_model_families) == 1 but len(allowed_component_graphs) > 1
    caplog.clear()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations,
        optimize_thresholds=False,
        allowed_model_families=[ModelFamily.LINEAR_MODEL],
        ensembling=True,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    pipeline_names = automl.rankings["pipeline_name"]
    assert pipeline_names.str.contains("Ensemble").any()
    assert (
        "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run."
        not in caplog.text
    )


def test_automl_max_iterations_less_than_ensembling_disabled(
    AutoMLTestEnv, X_y_binary, caplog
):
    max_iterations = _get_first_stacked_classifier_no([ModelFamily.LINEAR_MODEL])
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=max_iterations - 1,
        optimize_thresholds=False,
        allowed_model_families=[ModelFamily.LINEAR_MODEL],
        ensembling=True,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    assert (
        f"Ensembling is set to True, but max_iterations is too small, so ensembling will not run. Set max_iterations >= {max_iterations} to run ensembling."
        in caplog.text
    )

    pipeline_names = automl.rankings["pipeline_name"]
    assert not pipeline_names.str.contains("Ensemble").any()


def test_automl_max_batches_less_than_ensembling_disabled(
    AutoMLTestEnv, X_y_binary, caplog
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=2,
        optimize_thresholds=False,
        allowed_model_families=[ModelFamily.LINEAR_MODEL],
        ensembling=True,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    first_ensemble_batch = (
        1 + len(automl.allowed_pipelines) + 1
    )  # First batch + each pipeline batch
    assert (
        f"Ensembling is set to True, but max_batches is too small, so ensembling will not run. Set max_batches >= {first_ensemble_batch} to run ensembling."
        in caplog.text
    )

    pipeline_names = automl.rankings["pipeline_name"]
    assert not pipeline_names.str.contains("Ensemble").any()


@pytest.mark.parametrize("max_batches", [1, 2, 5, 10])
@pytest.mark.parametrize("verbose", [True, False])
def test_max_batches_output(max_batches, verbose, AutoMLTestEnv, X_y_binary, caplog):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=None,
        optimize_thresholds=False,
        max_batches=max_batches,
        verbose=verbose,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()

    output = caplog.text
    if verbose:
        assert output.count("Batch Number") == max_batches
    else:
        assert output.count("Batch Number") == 0


def test_max_batches_plays_nice_with_other_stopping_criteria(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary

    # Use the old default when all are None
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective="Log Loss Binary",
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    assert (
        len(automl.results["pipeline_results"])
        == len(get_estimators(problem_type="binary")) + 1
    )

    # Use max_iterations when both max_iterations and max_batches are set
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective="Log Loss Binary",
        max_batches=10,
        optimize_thresholds=False,
        max_iterations=6,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    assert len(automl.results["pipeline_results"]) == 6

    # Don't change max_iterations when only max_iterations is set
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=4,
        optimize_thresholds=False,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()
    assert len(automl.results["pipeline_results"]) == 4


@pytest.mark.parametrize("max_batches", [-1, -10, -np.inf])
def test_max_batches_must_be_non_negative(max_batches, X_y_binary):
    X, y = X_y_binary
    with pytest.raises(
        ValueError,
        match=f"Parameter max_batches must be None or non-negative. Received {max_batches}.",
    ):
        AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", max_batches=max_batches
        )


def test_stopping_criterion_bad(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(
        TypeError,
        match=r"Parameter max_time must be a float, int, string or None. Received <class 'tuple'> with value \('test',\).",
    ):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_time=("test",))
    with pytest.raises(
        ValueError,
        match=f"Parameter max_batches must be None or non-negative. Received -1.",
    ):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_batches=-1)
    with pytest.raises(
        ValueError,
        match=f"Parameter max_time must be None or non-negative. Received -1.",
    ):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_time=-1)
    with pytest.raises(
        ValueError,
        match=f"Parameter max_iterations must be None or non-negative. Received -1.",
    ):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", max_iterations=-1)


def test_data_splitter_binary(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary
    y[:] = 0
    y[0] = 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", n_jobs=1)
    env = AutoMLTestEnv("binary")
    with pytest.raises(Exception, match="Missing target values in the"):
        with pytest.warns(UserWarning):
            with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
                automl.search()

    y[1] = 1
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", n_jobs=1)
    with pytest.raises(Exception, match="Missing target values in the"):
        with pytest.warns(UserWarning):
            with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
                automl.search()

    y[2] = 1
    automl = AutoMLSearch(
        X_train=X, y_train=y, problem_type="binary", optimize_thresholds=False, n_jobs=1
    )
    with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
        automl.search()


def test_data_splitter_multi(AutoMLTestEnv, X_y_multi):
    X, y = X_y_multi
    y[:] = 1
    y[0] = 0

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="multiclass", n_jobs=1)
    env = AutoMLTestEnv("multiclass")
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
                automl.search()

    y[1] = 2
    # match based on regex, since data split doesn't have a random seed for reproducibility
    # regex matches the set {} and expects either 2 sets (missing in both train and test)
    #   or 1 set of multiple elements (both missing in train or both in test)
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="multiclass", n_jobs=1)
    with pytest.raises(Exception, match=r"(\{\d?\}.+\{\d?\})|(\{.+\,.+\})"):
        with pytest.warns(UserWarning):
            with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
                automl.search()

    y[1] = 0
    y[2:4] = 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="multiclass", n_jobs=1)
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
                automl.search()

    y[4] = 2
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="multiclass", n_jobs=1)
    with pytest.raises(Exception, match="Missing target values"):
        with pytest.warns(UserWarning):
            with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
                automl.search()

    y[5] = 0
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="multiclass", n_jobs=1)
    with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
        automl.search()


@patch("evalml.tuners.skopt_tuner.SKOptTuner.add")
def test_iterative_algorithm_pipeline_hyperparameters_make_pipeline_other_errors(
    mock_add, AutoMLTestEnv, X_y_multi
):
    X, y = X_y_multi
    custom_hyperparameters = {
        "Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", "mean"])}
    }
    estimators = get_estimators("multiclass", [ModelFamily.EXTRA_TREES])

    component_graphs = {}
    for ind, estimator in enumerate(estimators):
        component_graphs[f"CG_{ind}"] = [estimator]

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs=component_graphs,
        custom_hyperparameters=custom_hyperparameters,
        n_jobs=1,
    )
    env = AutoMLTestEnv("multiclass")

    mock_add.side_effect = ValueError("Alternate error that can be thrown")
    with pytest.raises(ValueError) as error:
        with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
            automl.search()
    assert "Alternate error that can be thrown" in str(error.value)
    assert "Default parameters for components" not in str(error.value)


@pytest.mark.parametrize("component_graphs", [True, False])
@pytest.mark.parametrize("automl_parameters", [True, False])
@pytest.mark.parametrize("custom_hyperparameters", [True, False])
def test_iterative_algorithm_pipeline_custom_hyperparameters_make_pipeline(
    custom_hyperparameters,
    automl_parameters,
    component_graphs,
    X_y_multi,
    AutoMLTestEnv,
):
    X, y = X_y_multi
    X = pd.DataFrame(X, columns=[f"Column_{i}" for i in range(20)])

    component_graph_ = None
    automl_parameters_ = None
    custom_hyperparameters_ = None

    if component_graphs:
        component_graph_ = {
            "Name_0": [
                "Drop Columns Transformer",
                "Imputer",
                "Random Forest Classifier",
            ]
        }

    if automl_parameters:
        automl_parameters_ = {
            "Drop Columns Transformer": {
                "columns": ["Column_0", "Column_1", "Column_2"]
            },
            "Random Forest Classifier": {"n_estimators": 201},
        }
    if custom_hyperparameters:
        custom_hyperparameters_ = {
            "Imputer": {"numeric_impute_strategy": Categorical(["mean"])},
            "Random Forest Classifier": {
                "max_depth": Integer(4, 7),
                "n_estimators": Integer(190, 210),
            },
        }

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs=component_graph_,
        pipeline_parameters=automl_parameters_,
        custom_hyperparameters=custom_hyperparameters_,
        max_batches=4,
    )
    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={"Log Loss Multiclass": 1.0}):
        automl.search()

    for i, row in automl.full_rankings.iterrows():
        if "Random Forest Classifier" in row["pipeline_name"]:
            if component_graph_ and automl_parameters:
                assert row["parameters"]["Drop Columns Transformer"]["columns"] == [
                    "Column_0",
                    "Column_1",
                    "Column_2",
                ]
            if custom_hyperparameters_:
                assert (
                    row["parameters"]["Imputer"]["numeric_impute_strategy"]
                    in custom_hyperparameters_["Imputer"]["numeric_impute_strategy"]
                )
                assert (
                    4 <= row["parameters"]["Random Forest Classifier"]["max_depth"] <= 7
                )
                if automl_parameters and row["id"] == 1:
                    assert (
                        row["parameters"]["Random Forest Classifier"]["n_estimators"]
                        == 201
                    )
                else:
                    assert (
                        190
                        <= row["parameters"]["Random Forest Classifier"]["n_estimators"]
                        <= 210
                    )
            else:
                assert row["parameters"]["Imputer"]["numeric_impute_strategy"] in [
                    "mean",
                    "median",
                    "most_frequent",
                ]
                assert (
                    1
                    <= row["parameters"]["Random Forest Classifier"]["max_depth"]
                    <= 10
                )
                if automl_parameters and row["id"] == 1:
                    assert (
                        row["parameters"]["Random Forest Classifier"]["n_estimators"]
                        == 201
                    )
                else:
                    assert (
                        10
                        <= row["parameters"]["Random Forest Classifier"]["n_estimators"]
                        <= 1000
                    )


def test_iterative_algorithm_passes_njobs_to_pipelines(
    dummy_classifier_estimator_class, X_y_binary, AutoMLTestEnv
):
    X, y = X_y_binary

    class MockEstimatorWithNJobs(Estimator):
        name = "Mock Classifier with njobs"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        hyperparameter_ranges = {}

        def __init__(self, n_jobs=-1, random_seed=0):
            super().__init__(
                parameters={"n_jobs": n_jobs},
                component_obj=None,
                random_seed=random_seed,
            )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        n_jobs=3,
        max_batches=2,
        allowed_component_graphs={
            "Pipeline 1": [MockEstimatorWithNJobs],
            "Pipeline 2": [MockEstimatorWithNJobs],
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.2}):
        automl.search()
    for parameters in automl.full_rankings.parameters:
        if "Mock Classifier with njobs" in parameters:
            assert parameters["Mock Classifier with njobs"]["n_jobs"] == 3
        else:
            assert all(
                "n_jobs" not in component_params
                for component_params in parameters.values()
            )


def test_automl_ensembling_false(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_time="60 seconds",
        max_batches=20,
        optimize_thresholds=False,
        ensembling=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.32}):
        automl.search()
    assert not automl.rankings["pipeline_name"].str.contains("Ensemble").any()


def test_search_with_text(AutoMLTestEnv):
    X = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
            ],
            "col_2": [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living",
                "Red, the blood of angry men - black, the dark of ages past",
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "Red, the blood of angry men - black, the dark of ages past",
                "It was red and yellow and green and brown and scarlet and black and ochre and peach and ruby and olive and violet and fawn...",
            ],
        }
    )
    X.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    y = [0, 1, 1, 0, 1, 0]
    automl = AutoMLSearch(
        X_train=X, y_train=y, problem_type="binary", optimize_thresholds=False
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl.rankings["pipeline_name"][1:].str.contains("Text").all()


@pytest.mark.parametrize(
    "problem_type,pipeline_name,ensemble_name",
    [
        (
            "binary",
            "Stacked Ensemble Classification Pipeline",
            "Stacked Ensemble Classifier",
        ),
        (
            "multiclass",
            "Stacked Ensemble Classification Pipeline",
            "Stacked Ensemble Classifier",
        ),
        (
            "regression",
            "Stacked Ensemble Regression Pipeline",
            "Stacked Ensemble Regressor",
        ),
    ],
)
@pytest.mark.parametrize("df_text", [True, False])
@patch("evalml.automl.automl_algorithm.IterativeAlgorithm.__init__")
def test_search_with_text_and_ensembling(
    mock_iter, df_text, problem_type, pipeline_name, ensemble_name
):
    X_with_text = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living",
                "Red, the blood of angry men - black, the dark of ages past",
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "Red, the blood of angry men - black, the dark of ages past",
                "It was red and yellow and green and brown and scarlet and black and ochre and peach and ruby and olive and violet and fawn...",
            ]
        }
    )
    X_no_text = pd.DataFrame({"col_1": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]})

    if df_text:
        X = X_with_text
        X.ww.init(logical_types={"col_1": "NaturalLanguage"})
    else:
        X = X_no_text
    if problem_type == "binary":
        y = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    elif problem_type == "multiclass":
        y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    else:
        y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    mock_iter.return_value = None
    _ = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        allowed_model_families=["random_forest", "decision_tree"],
        optimize_thresholds=False,
        max_batches=4,
        ensembling=True,
    )
    call_args = mock_iter.call_args_list[0][1]
    if df_text:
        assert call_args["text_in_ensembling"]
    else:
        assert not call_args["text_in_ensembling"]


def test_pipelines_per_batch(AutoMLTestEnv, X_y_binary):
    def total_pipelines(automl, num_batches, batch_size):
        total = 1 + len(automl.allowed_pipelines)
        total += (num_batches - 1) * batch_size
        return total

    X, y = X_y_binary

    # Checking for default of _pipelines_per_batch
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=2,
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl._pipelines_per_batch == 5
    assert automl._automl_algorithm.pipelines_per_batch == 5
    assert total_pipelines(automl, 2, 5) == len(automl.full_rankings)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=1,
        optimize_thresholds=False,
        _pipelines_per_batch=2,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl._pipelines_per_batch == 2
    assert automl._automl_algorithm.pipelines_per_batch == 2
    assert total_pipelines(automl, 1, 2) == len(automl.full_rankings)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=2,
        optimize_thresholds=False,
        _pipelines_per_batch=10,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl._pipelines_per_batch == 10
    assert automl._automl_algorithm.pipelines_per_batch == 10
    assert total_pipelines(automl, 2, 10) == len(automl.full_rankings)


def test_automl_respects_random_seed(
    AutoMLTestEnv, X_y_binary, dummy_classifier_estimator_class
):

    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs={"Name_0": [dummy_classifier_estimator_class]},
        optimize_thresholds=False,
        random_seed=42,
        max_iterations=10,
    )

    class DummyPipeline(BinaryClassificationPipeline):
        component_graph = [dummy_classifier_estimator_class]
        num_pipelines_different_seed = 0
        num_pipelines_init = 0

        def __init__(self, parameters, random_seed=0):
            is_diff_random_seed = not (random_seed == 42)
            self.__class__.num_pipelines_init += 1
            self.__class__.num_pipelines_different_seed += is_diff_random_seed
            super().__init__(
                self.component_graph, parameters=parameters, random_seed=random_seed
            )

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    pipelines = [DummyPipeline({}, random_seed=42)]
    automl._automl_algorithm = IterativeAlgorithm(
        max_iterations=2,
        allowed_pipelines=pipelines,
        tuner_class=SKOptTuner,
        random_seed=42,
        n_jobs=1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        pipeline_params={},
        custom_hyperparameters=None,
    )

    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl.allowed_pipelines[0].random_seed == 42
    assert (
        DummyPipeline.num_pipelines_different_seed == 0
        and DummyPipeline.num_pipelines_init
    )


@pytest.mark.parametrize(
    "callback", [log_error_callback, silent_error_callback, raise_error_callback]
)
@pytest.mark.parametrize("error_type", ["fit", "mean_cv_score", "fit-single"])
def test_automl_error_callback(error_type, callback, AutoMLTestEnv, X_y_binary, caplog):
    X, y = X_y_binary
    score_side_effect = None
    fit_side_effect = None
    score_return_value = {"Log Loss Binary": 0.8}
    if error_type == "mean_cv_score":
        msg = "Score Error!"
        score_side_effect = Exception(msg)
    elif error_type == "fit":
        msg = "all your model are belong to us"
        fit_side_effect = Exception(msg)
    else:
        # throw exceptions for only one pipeline
        msg = "all your model are belong to us"
        fit_side_effect = [Exception(msg)] * 3 + [None] * 100
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        error_callback=callback,
        train_best_pipeline=False,
        optimize_thresholds=False,
        n_jobs=1,
    )
    if callback in [log_error_callback, silent_error_callback]:
        exception = AutoMLSearchException
        match = "All pipelines in the current AutoML batch produced a score of np.nan on the primary objective"
    else:
        exception = Exception
        match = msg

    env = AutoMLTestEnv("binary")

    if error_type == "fit-single" and callback in [
        silent_error_callback,
        log_error_callback,
    ]:
        with env.test_context(
            mock_fit_side_effect=fit_side_effect,
            score_return_value=score_return_value,
            mock_score_side_effect=score_side_effect,
        ):
            automl.search()
    else:
        with pytest.raises(exception, match=match):
            with env.test_context(
                mock_fit_side_effect=fit_side_effect,
                score_return_value=score_return_value,
                mock_score_side_effect=score_side_effect,
            ):
                automl.search()

    if callback == silent_error_callback:
        assert msg not in caplog.text
    if callback == log_error_callback:
        assert f"Exception during automl search: {msg}" in caplog.text
        assert msg in caplog.text
    if callback in [raise_error_callback]:
        assert f"AutoML search raised a fatal exception: {msg}" in caplog.text
        assert msg in caplog.text


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_automl_woodwork_user_types_preserved(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    AutoMLTestEnv,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        score_return_value = {"Log Loss Binary": 1.0}

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        score_return_value = {"Log Loss Multiclass": 1.0}

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        score_return_value = {"R2": 1.0}

    X = pd.DataFrame(X)
    new_col = np.zeros(len(X))
    new_col[: int(len(new_col) / 2)] = 1
    X["cat col"] = pd.Series(new_col)
    X["num col"] = pd.Series(new_col)
    X["text col"] = pd.Series([f"{num}" for num in range(len(new_col))])
    X.ww.init(
        semantic_tags={"cat col": "category", "num col": "numeric"},
        logical_types={
            "cat col": "Categorical",
            "num col": "Integer",
            "text col": "NaturalLanguage",
        },
    )
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        max_batches=5,
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv(problem_type)
    with env.test_context(score_return_value=score_return_value):
        automl.search()
    for arg in env.mock_fit.call_args[0]:
        assert isinstance(arg, (pd.DataFrame, pd.Series))
        if isinstance(arg, pd.DataFrame):
            assert arg.ww.semantic_tags["cat col"] == {"category"}
            assert isinstance(
                arg.ww.logical_types["cat col"], ww.logical_types.Categorical
            )
            assert arg.ww.semantic_tags["num col"] == {"numeric"}
            assert isinstance(arg.ww.logical_types["num col"], ww.logical_types.Integer)
            assert arg.ww.semantic_tags["text col"] == set()
            assert isinstance(
                arg.ww.logical_types["text col"], ww.logical_types.NaturalLanguage
            )
    for arg in env.mock_score.call_args[0]:
        assert isinstance(arg, (pd.DataFrame, pd.Series))
        if isinstance(arg, pd.DataFrame):
            assert arg.ww.semantic_tags["cat col"] == {"category"}
            assert isinstance(
                arg.ww.logical_types["cat col"], ww.logical_types.Categorical
            )
            assert arg.ww.semantic_tags["num col"] == {"numeric"}
            assert isinstance(arg.ww.logical_types["num col"], ww.logical_types.Integer)
            assert arg.ww.semantic_tags["text col"] == set()
            assert isinstance(
                arg.ww.logical_types["text col"], ww.logical_types.NaturalLanguage
            )


def test_automl_validates_problem_configuration(ts_data):
    _, y = ts_data
    X = pd.DataFrame(pd.date_range("2020-10-01", "2020-10-31"), columns=["Date"])
    assert (
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary").problem_configuration
        == {}
    )
    assert (
        AutoMLSearch(
            X_train=X, y_train=y, problem_type="multiclass"
        ).problem_configuration
        == {}
    )
    assert (
        AutoMLSearch(
            X_train=X, y_train=y, problem_type="regression"
        ).problem_configuration
        == {}
    )
    msg = "user_parameters must be a dict containing values for at least the date_index, gap, max_delay, and forecast_horizon parameters"
    with pytest.raises(ValueError, match=msg):
        AutoMLSearch(X_train=X, y_train=y, problem_type="time series regression")
    with pytest.raises(ValueError, match=msg):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="time series regression",
            problem_configuration={"gap": 3},
        )
    with pytest.raises(ValueError, match=msg):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="time series regression",
            problem_configuration={"max_delay": 2, "gap": 3},
        )

    problem_config = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="time series regression",
        problem_configuration={
            "date_index": "Date",
            "max_delay": 2,
            "gap": 3,
            "forecast_horizon": 2,
        },
    ).problem_configuration
    assert problem_config == {
        "date_index": "Date",
        "max_delay": 2,
        "gap": 3,
        "forecast_horizon": 2,
    }


@patch("evalml.objectives.BinaryClassificationObjective.optimize_threshold")
def test_automl_best_pipeline(mock_optimize, X_y_binary):
    X, y = X_y_binary
    mock_optimize.return_value = 0.62

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        train_best_pipeline=False,
        n_jobs=1,
        max_iterations=3,
    )
    automl.search()
    with pytest.raises(PipelineNotYetFittedError, match="not fitted"):
        automl.best_pipeline.predict(X)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        objective="Accuracy Binary",
        n_jobs=1,
    )
    automl.search()
    automl.best_pipeline.predict(X)
    assert automl.best_pipeline.threshold == 0.5

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=True,
        objective="Log Loss Binary",
        n_jobs=1,
        max_iterations=3,
    )
    automl.search()
    automl.best_pipeline.predict(X)
    assert automl.best_pipeline.threshold == 0.62

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=True,
        objective="Accuracy Binary",
        n_jobs=1,
        max_iterations=3,
    )
    automl.search()
    automl.best_pipeline.predict(X)
    assert automl.best_pipeline.threshold == 0.62


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_automl_data_splitter_consistent(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    AutoMLTestEnv,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression

    data_splitters = []
    random_seed = [0, 0, 1]
    for seed in random_seed:
        a = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type=problem_type,
            random_seed=seed,
            optimize_thresholds=False,
            max_iterations=1,
        )
        env = AutoMLTestEnv(problem_type)
        with env.test_context():
            a.search()
        data_splitters.append(
            [[set(train), set(test)] for train, test in a.data_splitter.split(X, y)]
        )
    # append split from last random state again, should be referencing same datasplit object
    data_splitters.append(
        [[set(train), set(test)] for train, test in a.data_splitter.split(X, y)]
    )

    assert data_splitters[0] == data_splitters[1]
    assert data_splitters[1] != data_splitters[2]
    assert data_splitters[2] == data_splitters[3]


def test_automl_rerun(AutoMLTestEnv, X_y_binary, caplog):
    msg = "AutoMLSearch.search() has already been run and will not run again on the same instance"
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        train_best_pipeline=False,
        optimize_thresholds=False,
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 2}):
        automl.search()
    assert msg not in caplog.text
    with env.test_context(score_return_value={automl.objective.name: 2}):
        automl.search()
    assert msg in caplog.text


def test_timeseries_baseline_init_with_correct_gap_max_delay(
    AutoMLTestEnv, X_y_regression
):

    X, y = X_y_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="time series regression",
        problem_configuration={
            "date_index": None,
            "gap": 6,
            "max_delay": 3,
            "forecast_horizon": 7,
        },
        max_iterations=1,
    )
    env = AutoMLTestEnv("time series regression")
    with env.test_context():
        automl.search()

    # Best pipeline is baseline pipeline because we only run one iteration
    assert automl.best_pipeline.parameters == {
        "pipeline": {
            "date_index": None,
            "gap": 6,
            "max_delay": 0,
            "forecast_horizon": 7,
        },
        "Delayed Feature Transformer": {
            "date_index": None,
            "delay_features": False,
            "delay_target": True,
            "max_delay": 0,
            "gap": 6,
            "forecast_horizon": 7,
        },
        "Time Series Baseline Estimator": {"forecast_horizon": 7, "gap": 6},
    }


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.REGRESSION,
    ],
)
def test_automl_does_not_include_positive_only_objectives_by_default(
    problem_type, X_y_regression
):

    X, y = X_y_regression

    only_positive = []
    for name in get_all_objective_names():
        objective_class = get_objective(name)
        if objective_class.positive_only:
            only_positive.append(objective_class)

    search = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        problem_configuration={
            "date_index": None,
            "gap": 0,
            "max_delay": 0,
            "forecast_horizon": 2,
        },
    )
    assert search.objective not in only_positive
    assert all([obj not in only_positive for obj in search.additional_objectives])


@pytest.mark.parametrize("non_core_objective", get_non_core_objectives())
def test_automl_validate_objective(non_core_objective, X_y_regression):

    X, y = X_y_regression

    with pytest.raises(ValueError, match="is not allowed in AutoML!"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type=non_core_objective.problem_types[0],
            objective=non_core_objective.name,
        )

    with pytest.raises(ValueError, match="is not allowed in AutoML!"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type=non_core_objective.problem_types[0],
            additional_objectives=[non_core_objective.name],
        )


def test_automl_pipeline_params_simple(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary
    params = {
        "Imputer": {"numeric_impute_strategy": "most_frequent"},
        "Logistic Regression Classifier": {"C": 10, "penalty": "l2"},
        "Elastic Net Classifier": {"l1_ratio": 0.2},
    }
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        pipeline_parameters=params,
        optimize_thresholds=False,
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.23}):
        automl.search()
    for i, row in automl.rankings.iterrows():
        if "Imputer" in row["parameters"]:
            assert (
                row["parameters"]["Imputer"]["numeric_impute_strategy"]
                == "most_frequent"
            )
        if "Logistic Regression Classifier" in row["parameters"]:
            assert row["parameters"]["Logistic Regression Classifier"]["C"] == 10
            assert (
                row["parameters"]["Logistic Regression Classifier"]["penalty"] == "l2"
            )
        if "Elastic Net Classifier" in row["parameters"]:
            assert row["parameters"]["Elastic Net Classifier"]["l1_ratio"] == 0.2


def test_automl_pipeline_params_multiple(AutoMLTestEnv, X_y_regression):
    X, y = X_y_regression
    hyperparams = {
        "Imputer": {
            "numeric_impute_strategy": Categorical(["median", "most_frequent"])
        },
        "Decision Tree Regressor": {
            "max_depth": Categorical([17, 18, 19]),
            "max_features": Categorical(["auto"]),
        },
        "Elastic Net Regressor": {
            "alpha": Real(0, 0.5),
            "l1_ratio": Categorical((0.01, 0.02, 0.03)),
        },
    }
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        custom_hyperparameters=hyperparams,
        optimize_thresholds=False,
        n_jobs=1,
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 0.28}):
        automl.search()
    for i, row in automl.rankings.iterrows():
        if "Imputer" in row["parameters"]:
            assert row["parameters"]["Imputer"][
                "numeric_impute_strategy"
            ] == Categorical(["median", "most_frequent"]).rvs(
                random_state=automl.random_seed
            )
        if "Decision Tree Regressor" in row["parameters"]:
            assert row["parameters"]["Decision Tree Regressor"][
                "max_depth"
            ] == Categorical([17, 18, 19]).rvs(random_state=automl.random_seed)
            assert (
                row["parameters"]["Decision Tree Regressor"]["max_features"] == "auto"
            )
        if "Elastic Net Regressor" in row["parameters"]:
            assert 0 < row["parameters"]["Elastic Net Regressor"]["alpha"] < 0.5
            assert row["parameters"]["Elastic Net Regressor"][
                "l1_ratio"
            ] == Categorical((0.01, 0.02, 0.03)).rvs(random_state=automl.random_seed)


def test_automl_respects_pipeline_parameters_with_duplicate_components(
    AutoMLTestEnv, X_y_binary
):
    X, y = X_y_binary
    # Pass the input of the first imputer to the second imputer
    component_graph_dict = {
        "Imputer": ["Imputer", "X", "y"],
        "Imputer_1": ["Imputer", "Imputer.x", "y"],
        "Random Forest Classifier": ["Random Forest Classifier", "Imputer_1.x", "y"],
    }

    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        allowed_component_graphs={"Pipeline from dict": component_graph_dict},
        pipeline_parameters={
            "Imputer": {"numeric_impute_strategy": "most_frequent"},
            "Imputer_1": {"numeric_impute_strategy": "median"},
        },
        optimize_thresholds=False,
        max_batches=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 0.63}):
        automl.search()
    for row in automl.full_rankings.iloc[1:].parameters:
        assert row["Imputer"]["numeric_impute_strategy"] == "most_frequent"
        assert row["Imputer_1"]["numeric_impute_strategy"] == "median"

    component_graph_dict = {
        "One Hot Encoder": ["One Hot Encoder", "X", "y"],
        "One Hot Encoder_1": ["One Hot Encoder", "One Hot Encoder.x", "y"],
        "Random Forest Classifier": [
            "Random Forest Classifier",
            "One Hot Encoder_1.x",
            "y",
        ],
    }

    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        allowed_component_graphs={
            "Pipeline from dict": component_graph_dict,
        },
        pipeline_parameters={
            "One Hot Encoder": {"top_n": 15},
            "One Hot Encoder_1": {"top_n": 25},
        },
        optimize_thresholds=False,
        max_batches=1,
    )
    with env.test_context(score_return_value={automl.objective.name: 0.63}):
        automl.search()
    for row in automl.full_rankings.iloc[1:].parameters:
        assert row["One Hot Encoder"]["top_n"] == 15
        assert row["One Hot Encoder_1"]["top_n"] == 25


def test_automl_respects_pipeline_custom_hyperparameters_with_duplicate_components(
    AutoMLTestEnv, X_y_binary
):
    X, y = X_y_binary
    custom_hyperparameters = {
        "Imputer": {"numeric_impute_strategy": Categorical(["most_frequent", "mean"])},
        "Imputer_1": {"numeric_impute_strategy": Categorical(["median", "mean"])},
        "Random Forest Classifier": {"n_estimators": Categorical([50, 100])},
    }
    component_graph = {
        "Name_dict": {
            "Imputer": ["Imputer", "X", "y"],
            "Imputer_1": ["Imputer", "Imputer.x", "y"],
            "Random Forest Classifier": [
                "Random Forest Classifier",
                "Imputer_1.x",
                "y",
            ],
        }
    }

    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        allowed_component_graphs=component_graph,
        custom_hyperparameters=custom_hyperparameters,
        optimize_thresholds=False,
        max_batches=5,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 0.12}):
        automl.search()
    for i, row in automl.full_rankings.iterrows():
        if "Mode Baseline Binary" in row["pipeline_name"]:
            continue
        if row["pipeline_name"] == "Name_linear":
            assert row["parameters"]["Imputer"]["numeric_impute_strategy"] == "mean"
            assert row["parameters"]["Imputer_1"]["numeric_impute_strategy"] in {
                "most_frequent",
                "mean",
            }
            assert row["parameters"]["Random Forest Classifier"]["n_estimators"] in {
                100,
                125,
            }
        if row["pipeline_name"] == "Name_dict":
            assert row["parameters"]["Imputer"]["numeric_impute_strategy"] in {
                "most_frequent",
                "mean",
            }
            assert row["parameters"]["Imputer_1"]["numeric_impute_strategy"] in {
                "median",
                "mean",
            }
            assert row["parameters"]["Random Forest Classifier"]["n_estimators"] in {
                50,
                100,
            }


def test_automl_adds_pipeline_parameters_to_custom_pipeline_hyperparams(
    AutoMLTestEnv, X_y_binary
):
    X, y = X_y_binary

    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "Imputer_1": ["Imputer", "Imputer.x", "y"],
        "One Hot Encoder": ["One Hot Encoder", "Imputer_1.x", "y"],
        "Random Forest Classifier": [
            "Random Forest Classifier",
            "One Hot Encoder.x",
            "y",
        ],
    }

    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        allowed_component_graphs={
            "Pipeline One": component_graph,
            "Pipeline Two": component_graph,
            "Pipeline Three": component_graph,
        },
        pipeline_parameters={"Imputer": {"numeric_impute_strategy": "most_frequent"}},
        custom_hyperparameters={
            "One Hot Encoder": {"top_n": Categorical([12, 10])},
            "Imputer": {
                "numeric_impute_strategy": Categorical(["median", "most_frequent"])
            },
        },
        optimize_thresholds=False,
        max_batches=4,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.767}):
        automl.search()

    for i, row in automl.full_rankings.iterrows():
        if "Mode Baseline Binary" in row["pipeline_name"]:
            continue
        assert row["parameters"]["Imputer"]["numeric_impute_strategy"] in [
            "most_frequent",
            "median",
        ]
        assert 10 <= row["parameters"]["One Hot Encoder"]["top_n"] <= 12


def test_automl_pipeline_params_kwargs(AutoMLTestEnv, X_y_multi):
    X, y = X_y_multi
    hyperparams = {
        "Imputer": {"numeric_impute_strategy": Categorical(["most_frequent"])},
        "Decision Tree Classifier": {
            "max_depth": Integer(1, 2),
            "ccp_alpha": Real(0.1, 0.5),
        },
    }
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        custom_hyperparameters=hyperparams,
        allowed_model_families=[ModelFamily.DECISION_TREE],
        n_jobs=1,
    )
    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    for i, row in automl.rankings.iterrows():
        if "Imputer" in row["parameters"]:
            assert (
                row["parameters"]["Imputer"]["numeric_impute_strategy"]
                == "most_frequent"
            )
        if "Decision Tree Classifier" in row["parameters"]:
            assert (
                0.1 < row["parameters"]["Decision Tree Classifier"]["ccp_alpha"] < 0.5
            )
            assert row["parameters"]["Decision Tree Classifier"]["max_depth"] == 1


@pytest.mark.parametrize("random_seed", [0, 1, 9])
def test_automl_pipeline_random_seed(AutoMLTestEnv, random_seed, X_y_multi):
    X, y = X_y_multi
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        random_seed=random_seed,
        n_jobs=1,
    )
    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    for i, row in automl.rankings.iterrows():
        if "Base" not in list(row["parameters"].keys())[0]:
            assert automl.get_pipeline(row["id"]).random_seed == random_seed


@pytest.mark.parametrize(
    "ranges", [0, [float("-inf"), float("inf")], [float("-inf"), 0], [0, float("inf")]]
)
def test_automl_check_for_high_variance(
    ranges, X_y_binary, dummy_binary_pipeline_class
):
    X, y = X_y_binary
    if ranges == 0:
        objectives = "Log Loss Binary"
    else:
        objectives = CustomClassificationObjectiveRanges(ranges)
    automl = AutoMLSearch(
        X_train=X, y_train=y, problem_type="binary", objective=objectives
    )
    cv_scores = pd.Series([1, 1, 1])
    pipeline = dummy_binary_pipeline_class(parameters={})
    assert not automl._check_for_high_variance(pipeline, cv_scores)

    cv_scores = pd.Series([0, 0, 0])
    assert not automl._check_for_high_variance(pipeline, cv_scores)

    for cv_scores in [
        pd.Series([0, 1, np.nan, np.nan]),
        pd.Series([0, 1, 2, 3]),
        pd.Series([0, -1, -1, -1]),
        pd.Series([10, 0, -1, -10]),
    ]:
        if objectives == "Log Loss Binary":
            assert automl._check_for_high_variance(pipeline, cv_scores)
        else:
            assert not automl._check_for_high_variance(pipeline, cv_scores)


def test_automl_check_high_variance_logs_warning(AutoMLTestEnv, X_y_binary, caplog):
    X, y = X_y_binary

    env = AutoMLTestEnv("binary")

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
    )
    with env.test_context(score_return_value={"Log Loss Binary": 1}):
        automl.search()
    out = caplog.text
    assert "High coefficient of variation" not in out

    caplog.clear()

    desired_score_values = [{"Log Loss Binary": i} for i in [1, 2, 10] * 2]
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=2,
    )
    with env.test_context(mock_score_side_effect=desired_score_values):
        automl.search()
    out = caplog.text
    assert "High coefficient of variation" in out


def test_automl_raises_error_with_duplicate_pipeline_names(
    dummy_classifier_estimator_class, X_y_binary
):
    X, y = X_y_binary

    class MyPipeline(BinaryClassificationPipeline):
        estimator = dummy_classifier_estimator_class

    pipeline_0 = MyPipeline(
        custom_name="Custom Pipeline",
        component_graph=[dummy_classifier_estimator_class],
    )
    pipeline_1 = MyPipeline(
        custom_name="Custom Pipeline",
        component_graph=[dummy_classifier_estimator_class],
    )
    pipeline_2 = MyPipeline(
        custom_name="My Pipeline 3", component_graph=[dummy_classifier_estimator_class]
    )
    pipeline_3 = MyPipeline(
        custom_name="My Pipeline 3", component_graph=[dummy_classifier_estimator_class]
    )

    with pytest.raises(
        ValueError,
        match="All pipeline names must be unique. The name 'Custom Pipeline' was repeated.",
    ):
        AutoMLSearch(X, y, problem_type="binary",).train_pipelines(
            [
                pipeline_0,
                pipeline_1,
                pipeline_2,
            ],
        )

    with pytest.raises(
        ValueError,
        match="All pipeline names must be unique. The names 'Custom Pipeline', 'My Pipeline 3' were repeated.",
    ):
        AutoMLSearch(X, y, problem_type="binary",).train_pipelines(
            [
                pipeline_0,
                pipeline_1,
                pipeline_2,
                pipeline_3,
            ],
        )

    with pytest.raises(
        ValueError,
        match="All pipeline names must be unique. The names 'Custom Pipeline', 'My Pipeline 3' were repeated.",
    ):
        AutoMLSearch(X, y, problem_type="binary",).score_pipelines(
            [
                pipeline_0,
                pipeline_1,
                pipeline_2,
                pipeline_3,
            ],
            pd.DataFrame(),
            pd.Series(),
            None,
        )


def test_train_batch_score_batch(
    AutoMLTestEnv, dummy_binary_pipeline_class, X_y_binary
):
    def make_dummy_pipeline(index):
        class Pipeline(dummy_binary_pipeline_class):
            custom_name = f"Pipeline {index}"

        return Pipeline({})

    pipelines = [make_dummy_pipeline(i) for i in range(3)]

    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=3,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 0.9}):
        automl.search()

    with env.test_context(mock_fit_side_effect=[None, Exception("foo"), None]):
        fitted_pipelines = automl.train_pipelines(pipelines)
    assert fitted_pipelines.keys() == {"Pipeline 0", "Pipeline 2"}

    score_effects = [
        {"Log Loss Binary": 0.1},
        {"Log Loss Binary": 0.2},
        {"Log Loss Binary": 0.3},
    ]
    expected_scores = {
        f"Pipeline {i}": effect for i, effect in zip(range(3), score_effects)
    }
    with env.test_context(mock_score_side_effect=score_effects):
        scores = automl.score_pipelines(pipelines, X, y, ["Log Loss Binary"])
    assert scores == expected_scores


def test_train_batch_returns_trained_pipelines(X_y_binary):
    X, y = X_y_binary

    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")
    rf_pipeline = BinaryClassificationPipeline(
        ["Random Forest Classifier"],
        parameters={"Random Forest Classifier": {"n_jobs": 1}},
    )
    lrc_pipeline = BinaryClassificationPipeline(
        ["Logistic Regression Classifier"],
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
    )

    pipelines = [rf_pipeline, lrc_pipeline]
    fitted_pipelines = automl.train_pipelines(pipelines)

    assert all([isinstance(pl, PipelineBase) for pl in fitted_pipelines.values()])

    # Check that the output pipelines are fitted but the input pipelines are not
    for original_pipeline in pipelines:
        fitted_pipeline = fitted_pipelines[original_pipeline.name]
        assert fitted_pipeline.name == original_pipeline.name
        assert fitted_pipeline._is_fitted
        assert fitted_pipeline != original_pipeline
        assert fitted_pipeline.parameters == original_pipeline.parameters


@pytest.mark.parametrize(
    "pipeline_fit_side_effect",
    [
        [None] * 6,
        [None, Exception("foo"), None],
        [None, Exception("bar"), Exception("baz")],
        [Exception("Everything"), Exception("is"), Exception("broken")],
    ],
)
def test_train_batch_works(
    pipeline_fit_side_effect,
    AutoMLTestEnv,
    X_y_binary,
    dummy_binary_pipeline_class,
    stackable_classifiers,
    caplog,
):

    exceptions_to_check = [
        str(e) for e in pipeline_fit_side_effect if isinstance(e, Exception)
    ]

    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_time=1,
        max_iterations=2,
        train_best_pipeline=False,
        optimize_thresholds=False,
        n_jobs=1,
    )

    def make_pipeline_name(index):
        class DummyPipeline(dummy_binary_pipeline_class):
            custom_name = f"Pipeline {index}"

        return DummyPipeline({"Mock Classifier": {"a": index}})

    pipelines = [
        make_pipeline_name(i) for i in range(len(pipeline_fit_side_effect) - 1)
    ]
    input_pipelines = [
        BinaryClassificationPipeline([classifier])
        for classifier in stackable_classifiers[:2]
    ]
    ensemble = BinaryClassificationPipeline(
        [SklearnStackedEnsembleClassifier],
        parameters={
            "Sklearn Stacked Ensemble Classifier": {
                "input_pipelines": input_pipelines,
                "n_jobs": 1,
            }
        },
    )
    pipelines.append(ensemble)
    env = AutoMLTestEnv("binary")

    def train_batch_and_check():
        caplog.clear()
        with env.test_context(mock_fit_side_effect=pipeline_fit_side_effect):
            trained_pipelines = automl.train_pipelines(pipelines)

            assert len(trained_pipelines) == len(pipeline_fit_side_effect) - len(
                exceptions_to_check
            )
        assert env.mock_fit.call_count == len(pipeline_fit_side_effect)
        for exception in exceptions_to_check:
            assert exception in caplog.text

    # Test training before search is run
    train_batch_and_check()

    # Test training after search.
    with env.test_context(score_return_value={automl.objective.name: 1.2}):
        automl.search()
    train_batch_and_check()


no_exception_scores = {"F1": 0.9, "AUC": 0.7, "Log Loss Binary": 0.25}


@pytest.mark.parametrize(
    "pipeline_score_side_effect",
    [
        [no_exception_scores] * 6,
        [
            no_exception_scores,
            PipelineScoreError(
                exceptions={
                    "AUC": (Exception(), []),
                    "Log Loss Binary": (Exception(), []),
                },
                scored_successfully={"F1": 0.2},
            ),
            no_exception_scores,
        ],
        [
            no_exception_scores,
            PipelineScoreError(
                exceptions={
                    "AUC": (Exception(), []),
                    "Log Loss Binary": (Exception(), []),
                },
                scored_successfully={"F1": 0.3},
            ),
            PipelineScoreError(
                exceptions={"AUC": (Exception(), []), "F1": (Exception(), [])},
                scored_successfully={"Log Loss Binary": 0.2},
            ),
        ],
        [
            PipelineScoreError(
                exceptions={
                    "Log Loss Binary": (Exception(), []),
                    "F1": (Exception(), []),
                },
                scored_successfully={"AUC": 0.6},
            ),
            PipelineScoreError(
                exceptions={
                    "AUC": (Exception(), []),
                    "Log Loss Binary": (Exception(), []),
                },
                scored_successfully={"F1": 0.2},
            ),
            PipelineScoreError(
                exceptions={"Log Loss Binary": (Exception(), [])},
                scored_successfully={"AUC": 0.2, "F1": 0.1},
            ),
        ],
    ],
)
def test_score_batch_works(
    pipeline_score_side_effect,
    X_y_binary,
    dummy_classifier_estimator_class,
    AutoMLTestEnv,
    dummy_binary_pipeline_class,
    stackable_classifiers,
    caplog,
):

    exceptions_to_check = []
    expected_scores = {}
    for i, e in enumerate(pipeline_score_side_effect):
        # Ensemble pipeline has different name
        pipeline_name = (
            f"Pipeline {i}"
            if i < len(pipeline_score_side_effect) - 1
            else "Templated Pipeline"
        )
        scores = no_exception_scores
        if isinstance(e, PipelineScoreError):
            scores = {"F1": np.nan, "AUC": np.nan, "Log Loss Binary": np.nan}
            scores.update(e.scored_successfully)
            exceptions_to_check.append(f"Score error for {pipeline_name}")

        expected_scores[pipeline_name] = scores

    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class]
        },
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")

    def make_pipeline_name(index):
        class DummyPipeline(dummy_binary_pipeline_class):
            custom_name = f"Pipeline {index}"

        return DummyPipeline({"Mock Classifier": {"a": index}})

    pipelines = [
        make_pipeline_name(i) for i in range(len(pipeline_score_side_effect) - 1)
    ]
    input_pipelines = [
        BinaryClassificationPipeline([classifier])
        for classifier in stackable_classifiers[:2]
    ]
    ensemble = BinaryClassificationPipeline(
        [SklearnStackedEnsembleClassifier],
        parameters={
            "Sklearn Stacked Ensemble Classifier": {
                "input_pipelines": input_pipelines,
                "n_jobs": 1,
            }
        },
        custom_name="Templated Pipeline",
    )
    pipelines.append(ensemble)

    def score_batch_and_check():
        caplog.clear()
        with env.test_context(mock_score_side_effect=pipeline_score_side_effect):

            scores = automl.score_pipelines(
                pipelines, X, y, objectives=["Log Loss Binary", "F1", "AUC"]
            )
            assert scores == expected_scores
            for exception in exceptions_to_check:
                assert exception in caplog.text

    # Test scoring before search
    score_batch_and_check()

    with env.test_context(score_return_value={automl.objective.name: 3.12}):
        automl.search()

    # Test scoring after search
    score_batch_and_check()


def test_train_pipelines_score_pipelines_raise_exception_with_duplicate_names(
    X_y_binary, dummy_classifier_estimator_class, dummy_binary_pipeline_class
):
    class Pipeline1(dummy_binary_pipeline_class):
        custom_name = "My Pipeline"

    class Pipeline2(dummy_binary_pipeline_class):
        custom_name = "My Pipeline"

    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class]
        },
    )

    with pytest.raises(
        ValueError,
        match="All pipeline names must be unique. The name 'My Pipeline' was repeated.",
    ):
        automl.train_pipelines([Pipeline2({}), Pipeline1({})])

    with pytest.raises(
        ValueError,
        match="All pipeline names must be unique. The name 'My Pipeline' was repeated.",
    ):
        automl.score_pipelines([Pipeline2({}), Pipeline1({})], X, y, None)


def test_score_batch_before_fitting_yields_error_nan_scores(
    X_y_binary, dummy_classifier_estimator_class, dummy_binary_pipeline_class, caplog
):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class]
        },
    )

    scored_pipelines = automl.score_pipelines(
        [dummy_binary_pipeline_class({})], X, y, objectives=["Log Loss Binary", F1()]
    )
    assert scored_pipelines == {
        "Mock Binary Classification Pipeline": {"Log Loss Binary": np.nan, "F1": np.nan}
    }

    assert "Score error for Mock Binary Classification Pipeline" in caplog.text
    assert "This LabelEncoder instance is not fitted yet." in caplog.text


def test_high_cv_check_no_warning_for_divide_by_zero(
    X_y_binary, dummy_binary_pipeline_class
):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")
    with pytest.warns(None) as warnings:
        # mean is 0 but std is not
        automl._check_for_high_variance(
            dummy_binary_pipeline_class({}), cv_scores=[0.0, 1.0, -1.0]
        )
    assert len(warnings) == 0


@pytest.mark.parametrize(
    "automl_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS],
)
@patch("evalml.automl.engine.sequential_engine.train_pipeline")
def test_automl_supports_float_targets_for_classification(
    mock_train,
    automl_type,
    X_y_binary,
    X_y_multi,
    dummy_binary_pipeline_class,
    dummy_multiclass_pipeline_class,
    AutoMLTestEnv,
):
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        y = pd.Series(y).map({0: -5.19, 1: 6.7})
        mock_train.return_value = dummy_binary_pipeline_class({})
    else:
        X, y = X_y_multi
        y = pd.Series(y).map({0: -5.19, 1: 6.7, 2: 2.03})
        mock_train.return_value = dummy_multiclass_pipeline_class({})

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        random_seed=0,
        n_jobs=1,
    )
    env = AutoMLTestEnv(automl.problem_type)
    with env.test_context(score_return_value={automl.objective.name: 0.1}):
        automl.search()

    # Assert that we train pipeline on the original target, not the encoded one used in EngineBase for data splitting
    _, kwargs = mock_train.call_args
    mock_y = kwargs["y"]
    pd.testing.assert_series_equal(mock_y, y, check_dtype=False)


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ],
)
def test_automl_issues_beta_warning_for_time_series(problem_type, X_y_binary):

    X, y = X_y_binary

    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter("always")
        AutoMLSearch(
            X,
            y,
            problem_type=problem_type,
            problem_configuration={
                "date_index": None,
                "gap": 0,
                "max_delay": 2,
                "forecast_horizon": 9,
            },
        )
        assert len(warn) == 1
        message = "Time series support in evalml is still in beta, which means we are still actively building its core features"
        assert str(warn[0].message).startswith(message)


def test_automl_drop_index_columns(AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X["index_col"] = pd.Series(range(len(X)))
    X.ww.init(index="index_col")

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_batches=2,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    for pipeline in automl.allowed_pipelines:
        assert pipeline.get_component("Drop Columns Transformer")
        assert "Drop Columns Transformer" in pipeline.parameters
        assert pipeline.parameters["Drop Columns Transformer"] == {
            "columns": ["index_col"]
        }

    all_drop_column_params = []
    for _, row in automl.full_rankings.iterrows():
        if "Baseline" not in row.pipeline_name:
            all_drop_column_params.append(
                row.parameters["Drop Columns Transformer"]["columns"]
            )
    assert all(param == ["index_col"] for param in all_drop_column_params)


def test_automl_validates_data_passed_in_to_allowed_component_graphs(
    X_y_binary, dummy_classifier_estimator_class, dummy_binary_pipeline_class
):
    X, y = X_y_binary

    with pytest.raises(
        ValueError,
        match="Parameter allowed_component_graphs must be either None or a dictionary!",
    ):
        AutoMLSearch(
            X,
            y,
            problem_type="binary",
            allowed_component_graphs=[
                {
                    "Mock Binary Classification Pipeline": [
                        dummy_classifier_estimator_class
                    ]
                }
            ],
        )

    with pytest.raises(
        ValueError,
        match="Every component graph passed must be of type list, dictionary, or ComponentGraph!",
    ):
        AutoMLSearch(
            X,
            y,
            problem_type="binary",
            allowed_component_graphs={
                "Mock Binary Classification Pipeline": dummy_classifier_estimator_class
            },
        )


@pytest.mark.parametrize(
    "problem_type",
    [
        problem_type
        for problem_type in ProblemTypes.all_problem_types
        if not is_time_series(problem_type)
    ],
)
def test_automl_baseline_pipeline_predictions_and_scores(problem_type):
    X = pd.DataFrame({"one": [1, 2, 3, 4], "two": [2, 3, 4, 5], "three": [1, 2, 3, 4]})
    y = pd.Series([10, 11, 10, 10])
    if problem_type == ProblemTypes.MULTICLASS:
        y = pd.Series([10, 11, 12, 11])
    automl = AutoMLSearch(X, y, problem_type=problem_type)
    baseline = automl._get_baseline_pipeline()
    baseline.fit(X, y)

    if problem_type == ProblemTypes.BINARY:
        expected_predictions = pd.Series(np.array([10] * len(X)), dtype="int64")
        expected_predictions_proba = pd.DataFrame(
            {10: [1.0, 1.0, 1.0, 1.0], 11: [0.0, 0.0, 0.0, 0.0]}
        )
    if problem_type == ProblemTypes.MULTICLASS:
        expected_predictions = pd.Series(np.array([11] * len(X)), dtype="int64")
        expected_predictions_proba = pd.DataFrame(
            {
                10: [0.0, 0.0, 0.0, 0.0],
                11: [1.0, 1.0, 1.0, 1.0],
                12: [0.0, 0.0, 0.0, 0.0],
            }
        )
    if problem_type == ProblemTypes.REGRESSION:
        mean = y.mean()
        expected_predictions = pd.Series([mean] * len(X))

    pd.testing.assert_series_equal(expected_predictions, baseline.predict(X))
    if is_classification(problem_type):
        pd.testing.assert_frame_equal(
            expected_predictions_proba, baseline.predict_proba(X)
        )
    np.testing.assert_allclose(
        baseline.feature_importance.iloc[:, 1], np.array([0.0] * X.shape[1])
    )


@pytest.mark.parametrize(
    "problem_type",
    [
        problem_type
        for problem_type in ProblemTypes.all_problem_types
        if is_time_series(problem_type)
    ],
)
def test_automl_baseline_pipeline_predictions_and_scores_time_series(problem_type):
    X = pd.DataFrame({"a": [4, 5, 6, 7, 8]})
    y = pd.Series([0, 1, 1, 0, 1])
    expected_predictions_proba = pd.DataFrame(
        {
            0: pd.Series([1.0], index=[4]),
            1: pd.Series([0.0], index=[4]),
        }
    )
    if problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        y = pd.Series([0, 2, 0, 1, 1])
        expected_predictions_proba = pd.DataFrame(
            {
                0: pd.Series([0.0], index=[4]),
                1: pd.Series([1.0], index=[4]),
                2: pd.Series([0.0], index=[4]),
            }
        )

    automl = AutoMLSearch(
        X,
        y,
        problem_type=problem_type,
        problem_configuration={
            "date_index": None,
            "gap": 0,
            "max_delay": 1,
            "forecast_horizon": 1,
        },
    )
    baseline = automl._get_baseline_pipeline()
    X_train, y_train = X[:4], y[:4]
    X_validation = X[4:]
    baseline.fit(X_train, y_train)

    expected_predictions = y.shift(1)[4:]
    if problem_type != ProblemTypes.TIME_SERIES_REGRESSION:
        expected_predictions = expected_predictions.astype("int64")
    preds = baseline.predict(X_validation, None, X_train, y_train)
    pd.testing.assert_series_equal(expected_predictions, preds)
    if is_classification(problem_type):
        pd.testing.assert_frame_equal(
            expected_predictions_proba,
            baseline.predict_proba(X_validation, X_train, y_train),
        )
    np.testing.assert_allclose(
        baseline.feature_importance.iloc[:, 1], np.array([0.0] * X_validation.shape[1])
    )


@pytest.mark.parametrize(
    "objective,errors",
    [
        ("Log Loss Binary", True),
        ("AUC", True),
        ("F1", False),
        ("Accuracy Binary", False),
    ],
)
@pytest.mark.parametrize("verbose", [True, False])
@patch(
    "evalml.objectives.binary_classification_objective.BinaryClassificationObjective.optimize_threshold",
    return_value=0.65,
)
def test_automl_alternate_thresholding_objective(
    mock_optimize,
    objective,
    errors,
    verbose,
    X_y_binary,
    caplog,
):
    X, y = X_y_binary
    if errors:
        with pytest.raises(
            ValueError,
            match="Alternate thresholding objective must be a tuneable objective",
        ):
            AutoMLSearch(
                X_train=X,
                y_train=y,
                problem_type="binary",
                alternate_thresholding_objective=objective,
            )
        return
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        alternate_thresholding_objective=objective,
        max_iterations=1,
        verbose=verbose,
    )
    automl.search()
    mock_optimize.assert_called()
    assert ("Optimal threshold found" in caplog.text) == verbose
    assert automl.best_pipeline.threshold == 0.65


@pytest.mark.parametrize("threshold", [False, True])
@patch("evalml.objectives.standard_metrics.F1.optimize_threshold", return_value=0.42)
def test_automl_thresholding_train_pipelines(mock_objective, threshold, X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=threshold,
        max_iterations=1,
    )
    automl.search()
    if threshold:
        mock_objective.assert_called()
        assert automl.best_pipeline.threshold == 0.42
    else:
        mock_objective.assert_not_called()
        assert automl.best_pipeline.threshold is None
    pipelines_to_train = [automl.get_pipeline(0)]
    pipes = automl.train_pipelines(pipelines_to_train)
    if threshold:
        mock_objective.assert_called()
        assert all([p.threshold == 0.42 for p in pipes.values()])
    else:
        mock_objective.assert_not_called()
        assert all([p.threshold is None for p in pipes.values()])


@pytest.mark.parametrize("columns", [[], ["unknown_col"], ["unknown1, unknown2"]])
def test_automl_drop_unknown_columns(columns, AutoMLTestEnv, X_y_binary, caplog):
    caplog.clear()
    X, y = X_y_binary
    X = pd.DataFrame(X)
    for col in columns:
        X[col] = pd.Series(range(len(X)))
    X.ww.init()
    X.ww.set_types({col: "Unknown" for col in columns})
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_batches=2,
        verbose=True,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    if not len(columns):
        for pipeline in automl.allowed_pipelines:
            assert "Drop Columns Transformer" not in pipeline.name
        assert "because they are of 'Unknown'" not in caplog.text
        return

    assert "because they are of 'Unknown'" in caplog.text
    for pipeline in automl.allowed_pipelines:
        assert pipeline.get_component("Drop Columns Transformer")
        assert "Drop Columns Transformer" in pipeline.parameters
        assert pipeline.parameters["Drop Columns Transformer"] == {"columns": columns}

    all_drop_column_params = []
    for _, row in automl.full_rankings.iterrows():
        if "Baseline" not in row.pipeline_name:
            all_drop_column_params.append(
                row.parameters["Drop Columns Transformer"]["columns"]
            )
    assert all(param == columns for param in all_drop_column_params)


@pytest.mark.parametrize(
    "automl_type",
    [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ],
)
def test_data_splitter_gives_pipelines_same_data(
    automl_type, AutoMLTestEnv, X_y_binary, X_y_multi, X_y_regression
):
    problem_configuration = None
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
    elif automl_type == ProblemTypes.TIME_SERIES_REGRESSION:
        problem_configuration = {
            "gap": 1,
            "max_delay": 1,
            "date_index": 0,
            "forecast_horizon": 10,
        }
        X, y = X_y_regression
    else:
        problem_configuration = {
            "gap": 1,
            "max_delay": 1,
            "date_index": 0,
            "forecast_horizon": 10,
        }
        X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        max_batches=1,
        n_jobs=1,
        problem_configuration=problem_configuration,
    )
    n_splits = automl.data_splitter.n_splits
    env = AutoMLTestEnv(automl_type)
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    n_pipelines_evaluated = len(automl.results["pipeline_results"])
    assert n_pipelines_evaluated > 1

    # current automl algo trains each pipeline using 3-fold CV for "small" datasets (i.e. test data above)
    # therefore, each pipeline should recieve an identical set of three training-validation splits
    X_fit_hashes, y_fit_hashes = defaultdict(set), defaultdict(set)
    X_score_hashes, y_score_hashes = defaultdict(set), defaultdict(set)
    for evaluation_index in range(0, n_pipelines_evaluated * n_splits, n_splits):
        for fold_index in range(n_splits):
            fold_fit_X, fold_fit_y = env.mock_fit.call_args_list[
                evaluation_index + fold_index
            ][0]
            fold_score_X, fold_score_y = env.mock_score.call_args_list[
                evaluation_index + fold_index
            ][0]
            X_fit_hashes[fold_index].add(joblib_hash(fold_fit_X))
            y_fit_hashes[fold_index].add(joblib_hash(fold_fit_y))
            X_score_hashes[fold_index].add(joblib_hash(fold_score_X))
            y_score_hashes[fold_index].add(joblib_hash(fold_score_y))

    for data_hash_dictionary in [
        X_fit_hashes,
        y_fit_hashes,
        X_score_hashes,
        y_score_hashes,
    ]:
        assert (
            len(data_hash_dictionary) == n_splits
        ), f"We should have hashes for exactly {n_splits} splits"
        assert all(
            len(data_hash_dictionary[i]) == 1 for i in range(n_splits)
        ), "There should only be one hash per split."


@pytest.mark.parametrize(
    "allowed_component_graphs",
    [None, {"graph": ["Imputer", "Logistic Regression Classifier"]}],
)
@pytest.mark.parametrize(
    "pipeline_parameters,set_values",
    [
        ({"Logistic Regression Classifier": {"penalty": "l1"}}, {}),
        (
            {
                "Imputer": {"numeric_impute_strategy": "mean"},
                "Logistic Regression Classifier": {"penalty": "l1"},
            },
            {},
        ),
        (
            {
                "Undersampler": {"sampling_ratio": 0.05},
                "Logistic Regression Classifier": {"penalty": "l1"},
            },
            {"Undersampler"},
        ),
        (
            {
                "Undersampler": {"sampling_ratio": 0.05},
                "Oversampler": {"sampling_ratio": 0.10},
            },
            {"Undersampler", "Oversampler"},
        ),
    ],
)
def test_pipeline_parameter_warnings_component_graphs(
    pipeline_parameters, set_values, allowed_component_graphs, AutoMLTestEnv, X_y_binary
):
    X, y = X_y_binary
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(
            "always",
            category=ParameterNotUsedWarning,
        )
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            max_batches=2,
            n_jobs=1,
            allowed_component_graphs=allowed_component_graphs,
            pipeline_parameters=pipeline_parameters,
        )
        env = AutoMLTestEnv("binary")
        with env.test_context(score_return_value={automl.objective.name: 1.0}):
            automl.search()
    assert len(w) == (1 if len(set_values) else 0)
    if len(w):
        assert w[0].message.components == set_values


@patch("evalml.pipelines.utils._get_preprocessing_components")
@pytest.mark.parametrize("verbose", [True, False])
def test_pipeline_parameter_warnings_with_other_types(
    mock_get_preprocessing_components, verbose, X_y_regression
):
    X, y = X_y_regression

    def dummy_mock_get_preprocessing_components(*args, **kwargs):
        warnings.warn(UserWarning("dummy test warning"))
        return ["Imputer"]

    mock_get_preprocessing_components.side_effect = (
        dummy_mock_get_preprocessing_components
    )
    with pytest.warns(None) as warnings_logged:
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
            allowed_model_families=["random_forest", "linear_model"],
            pipeline_parameters={"Decision Tree Classifier": {"max_depth": 1}},
            verbose=verbose,
        )

    assert len(warnings_logged) == 2
    assert isinstance(warnings_logged[0].message, UserWarning)
    assert isinstance(warnings_logged[1].message, ParameterNotUsedWarning)


@pytest.mark.parametrize("nans", [None, pd.NA, np.nan])
@patch("evalml.pipelines.components.Estimator.fit")
@patch(
    "evalml.pipelines.BinaryClassificationPipeline.score",
    return_value={"Log Loss Binary": 0.5},
)
def test_search_with_text_nans(mock_score, mock_fit, nans):
    X = pd.DataFrame(
        {
            "a": [np.nan] + [i for i in range(99)],
            "b": [np.nan] + [f"string {i} is valid" for i in range(99)],
        }
    )
    X.ww.init(logical_types={"b": "NaturalLanguage"})
    y = pd.Series([0] * 25 + [1] * 75)
    automl = AutoMLSearch(
        X_train=X, y_train=y, problem_type="binary", optimize_thresholds=False
    )
    automl.search()
    for (x, _), _ in mock_fit.call_args_list:
        assert all(
            [str(types) == "Double" for types in x.ww.types["Logical Type"].values]
        )


@pytest.mark.parametrize(
    "engine_str",
    engine_strs + ["sequential", "cf_process", "invalid option"],
)
def test_build_engine(engine_str):
    """Test to ensure that AutoMLSearch's build_engine_from_str() chooses
    and returns an instance of the correct engine."""
    if "cf" in engine_str:
        expected_engine_type = CFEngine
        engine = build_engine_from_str(engine_str)
        assert isinstance(engine, expected_engine_type)
        engine.close()
    elif "dask" in engine_str:
        expected_engine_type = DaskEngine
        engine = build_engine_from_str(engine_str)
        assert isinstance(engine, expected_engine_type)
        engine.close()
    elif "sequential" in engine_str:
        expected_engine_type = SequentialEngine
        engine = build_engine_from_str(engine_str)
        assert isinstance(engine, expected_engine_type)
        engine.close()
    else:
        with pytest.raises(
            ValueError, match="is not a valid engine, please choose from"
        ):
            build_engine_from_str(engine_str)


@pytest.mark.parametrize(
    "engine_choice",
    ["str", "engine_instance", "invalid_type", "invalid_str"],
)
def test_automl_chooses_engine(engine_choice, X_y_binary):
    """Test that ensures that AutoMLSearch chooses an engine for valid input types and raises
    the proper exception for others."""
    X, y = X_y_binary
    if engine_choice == "str":
        engine_choice = "dask_process"
        automl = AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", engine=engine_choice
        )
        assert isinstance(automl._engine, DaskEngine)
        automl.close_engine()
    elif engine_choice == "engine_instance":
        engine_choice = DaskEngine()
        automl = AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", engine=engine_choice
        )
        automl.close_engine()
    elif engine_choice == "invalid_str":
        engine_choice = "DaskEngine"
        with pytest.raises(
            ValueError, match="is not a valid engine, please choose from"
        ):
            automl = AutoMLSearch(
                X_train=X, y_train=y, problem_type="binary", engine=engine_choice
            )
    elif engine_choice == "invalid_type":
        engine_choice = DaskEngine
        with pytest.raises(
            TypeError,
            match="Invalid type provided for 'engine'.  Requires string, DaskEngine instance, or CFEngine instance.",
        ):
            automl = AutoMLSearch(
                X_train=X, y_train=y, problem_type="binary", engine=engine_choice
            )


@patch("evalml.pipelines.components.RandomForestClassifier.predict_proba")
@patch("evalml.pipelines.components.RandomForestClassifier.predict")
@patch("evalml.pipelines.components.ElasticNetClassifier.predict_proba")
@patch("evalml.pipelines.components.ElasticNetClassifier.predict")
def test_automl_ensembler_allowed_component_graphs(
    mock_en_predict,
    mock_en_predict_proba,
    mock_rf_predict,
    mock_rf_predict_proba,
    X_y_regression,
    caplog,
):
    """
    Test that graphs defined in allowed_component_graphs are able to be put in an ensemble pipeline.
    """
    X, y = X_y_regression
    mock_en_predict_proba.return_value = np.ones(len(y))
    mock_rf_predict_proba.return_value = np.ones(len(y))
    mock_en_predict.return_value = np.ones(len(y))
    mock_rf_predict.return_value = np.ones(len(y))
    component_graphs = {
        "Pipeline1": {
            "Imputer": ["Imputer", "X", "y"],
            "Log Transformer": ["Log Transformer", "X", "y"],
            "RF": ["Random Forest Regressor", "Imputer.x", "Log Transformer.y"],
        },
        "Pipeline2": {
            "Imputer": ["Imputer", "X", "y"],
            "Log Transformer": ["Log Transformer", "X", "y"],
            "EN": ["Elastic Net Regressor", "Imputer.x", "Log Transformer.y"],
        },
    }
    automl = AutoMLSearch(
        X,
        y,
        "regression",
        allowed_component_graphs=component_graphs,
        ensembling=True,
        max_batches=4,
        verbose=True,
    )
    automl.search()
    assert "Stacked Ensemble Regression Pipeline" in caplog.text
    assert "Stacked Ensemble Regression Pipeline" in list(
        automl.rankings["pipeline_name"]
    )
    ensemble_result = automl.rankings[
        automl.rankings["pipeline_name"] == "Stacked Ensemble Regression Pipeline"
    ]
    assert not np.isnan(float(ensemble_result["mean_cv_score"]))


@pytest.mark.parametrize(
    "automl_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_baseline_pipeline_properly_initalized(
    automl_type,
    AutoMLTestEnv,
    X_y_binary,
    X_y_multi,
    X_y_regression,
):
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        score_value = {"Log Loss Binary": 1.0}
        expected_pipeline = BinaryClassificationPipeline(
            component_graph=["Baseline Classifier"],
            custom_name="Mode Baseline Binary Classification Pipeline",
            parameters={"Baseline Classifier": {"strategy": "mode"}},
        )
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        score_value = {"Log Loss Multiclass": 1.0}
        expected_pipeline = MulticlassClassificationPipeline(
            component_graph=["Baseline Classifier"],
            custom_name="Mode Baseline Multiclass Classification Pipeline",
            parameters={"Baseline Classifier": {"strategy": "mode"}},
        )
    elif automl_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        score_value = {"R2": 1.0}
        expected_pipeline = RegressionPipeline(
            component_graph=["Baseline Regressor"],
            custom_name="Mean Baseline Regression Pipeline",
            parameters={"Baseline Regressor": {"strategy": "mean"}},
        )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=automl_type,
        optimize_thresholds=False,
        max_iterations=1,
    )
    env = AutoMLTestEnv(automl_type)
    with env.test_context(score_return_value=score_value):
        automl.search()

    baseline_pipeline = automl.get_pipeline(0)
    assert expected_pipeline == baseline_pipeline
