import inspect
import os
import warnings
from collections import OrderedDict, defaultdict
from itertools import product
from math import ceil
from unittest.mock import MagicMock, PropertyMock, patch

import cloudpickle
import featuretools as ft
import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from joblib import hash as joblib_hash
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn import datasets
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
    MulticlassClassificationPipeline,
    PipelineBase,
    RegressionPipeline,
    StackedEnsembleClassifier,
)
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    DecisionTreeClassifier,
    EmailFeaturizer,
    NaturalLanguageFeaturizer,
    TimeSeriesFeaturizer,
    URLFeaturizer,
)
from evalml.pipelines.utils import (
    _get_pipeline_base_class,
    _make_stacked_ensemble_pipeline,
)
from evalml.preprocessing import TimeSeriesSplit, TrainingValidationSplit, split_data
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_classification,
    is_time_series,
)
from evalml.tests.automl_tests.parallel_tests.test_automl_dask import engine_strs
from evalml.tests.automl_tests.test_automl_iterative_algorithm import (
    _get_first_stacked_classifier_no,
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
            == pd.Series([fold["mean_cv_score"] for fold in results["cv_data"]]).mean()
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
                "validation_score",
                "mean_cv_score",
                "standard_deviation_cv_score",
                "percent_better_than_baseline",
                "high_variance_cv",
                "parameters",
            ],
        ),
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
                "validation_score",
                "mean_cv_score",
                "standard_deviation_cv_score",
                "percent_better_than_baseline",
                "high_variance_cv",
                "parameters",
            ],
        ),
    )


def test_search_batch_times(caplog, X_y_binary, AutoMLTestEnv):
    caplog.clear()
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=None,
        optimize_thresholds=False,
        max_batches=3,
        verbose=True,
        timing=True,
    )
    batch_times = None
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        batch_times = automl.search()

    out = caplog.text
    assert isinstance(batch_times, dict)
    assert isinstance(list(batch_times.keys())[0], int)
    assert isinstance(batch_times[1], dict)
    assert isinstance(list(batch_times[1].keys())[0], str)
    assert isinstance(batch_times[1]["Total time of batch"], str)
    assert isinstance(batch_times[2]["Total time of batch"], str)
    assert isinstance(batch_times[3]["Total time of batch"], str)

    assert len(batch_times) == 3
    assert len(batch_times[1]) == 3
    assert len(batch_times[2]) == 3
    assert len(batch_times[3]) == 7

    assert "Batch Time Stats" in out
    assert "Batch 1 time stats" in out
    assert "Batch 2 time stats" in out
    assert "Batch 3 time stats" in out


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
        assert "Using default limit of max_batches=3." in out
        assert "Searching up to 3 batches for a total of" in out
    else:
        assert "Using default limit of max_batches=3." not in out
        assert "Searching up to 3 batches for a total of" not in out
    assert len(automl.results["pipeline_results"]) > 0

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
        mock_fit_side_effect=Exception("all your model are belong to us"),
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
        mock_score_side_effect=Exception("all your model are belong to us"),
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
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_iterations=3,
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.03}):
        automl.search()
    assert len(automl.full_rankings) > 0
    assert len(automl.rankings) > 0
    assert len(automl.full_rankings) >= len(automl.rankings)

    X, y = X_y_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        max_iterations=3,
        optimize_thresholds=False,
        n_jobs=1,
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={"R2": 0.03}):
        automl.search()
    assert len(automl.full_rankings) > 0
    assert len(automl.rankings) > 0
    assert len(automl.full_rankings) >= len(automl.rankings)


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


@patch("evalml.tuners.random_search_tuner.RandomSearchTuner.is_search_space_exhausted")
def test_automl_tuner_exception(
    mock_is_search_space_exhausted,
    AutoMLTestEnv,
    X_y_binary,
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
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("binary")
    with pytest.raises(NoParamsException, match=error_text):
        with env.test_context(score_return_value={"Log Loss Binary": 0.2}):
            automl.search()


@patch("evalml.automl.automl_algorithm.DefaultAlgorithm.next_batch")
def test_automl_algorithm(
    mock_algo_next_batch,
    AutoMLTestEnv,
    X_y_binary,
):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=5,
        automl_algorithm="default",
    )
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
            interactive_plot=True,
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
                                value,
                                loaded_[name][objective_name],
                            )
                    elif name == "percent_better_than_baseline":
                        np.testing.assert_almost_equal(
                            pipeline_results[name],
                            loaded_[name],
                        )
                    else:
                        assert pipeline_results[name] == loaded_[name]

        pd.testing.assert_frame_equal(automl.rankings, loaded_automl.rankings)


@patch("cloudpickle.dump")
def test_automl_serialization_protocol(mock_cloudpickle_dump, tmpdir, X_y_binary):
    X, y = X_y_binary
    path = os.path.join(str(tmpdir), "automl.pkl")
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=5,
        n_jobs=1,
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
            X_train=X,
            y_train=y,
            problem_type="binary",
            data_splitter=data_splitter,
        )


def test_large_dataset_binary(AutoMLTestEnv):
    X = pd.DataFrame(
        {"col_0": range(111113)},
    )  # Reaches just over max row threshold after holdout set
    y = pd.Series([i % 2 for i in range(111113)])

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
            automl.results["pipeline_results"][pipeline_id]["validation_score"] == 1.234
        )
        assert np.isnan(
            automl.results["pipeline_results"][pipeline_id]["mean_cv_score"],
        )


def test_large_dataset_multiclass(AutoMLTestEnv):
    X = pd.DataFrame(
        {"col_0": range(111113)},
    )  # Reaches just over max row threshold after holdout set
    y = pd.Series([i % 4 for i in range(111113)])

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
            automl.results["pipeline_results"][pipeline_id]["validation_score"] == 1.234
        )
        assert np.isnan(
            automl.results["pipeline_results"][pipeline_id]["mean_cv_score"],
        )


def test_large_dataset_regression(AutoMLTestEnv):
    X = pd.DataFrame(
        {"col_0": [i for i in range(200000)]},
    )  # Reaches just over max row threshold after holdout set
    y = pd.Series([i for i in range(200000)])

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
            automl.results["pipeline_results"][pipeline_id]["validation_score"] == 1.234
        )
        assert np.isnan(
            automl.results["pipeline_results"][pipeline_id]["mean_cv_score"],
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

    # under_max_rows = (
    #     _LARGE_DATA_ROW_THRESHOLD + int(ceil(_LARGE_DATA_ROW_THRESHOLD * 0.1 / 0.9)) - 1
    # )  # Should be under threshold even after taking out holdout set
    X, y = generate_fake_dataset(_LARGE_DATA_ROW_THRESHOLD)
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
    over_max_rows = (
        _LARGE_DATA_ROW_THRESHOLD + int(ceil(_LARGE_DATA_ROW_THRESHOLD * 0.1 / 0.9)) + 1
    )
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
        automl.results["pipeline_results"][0]["mean_cv_score"],
        0.0,
        decimal=4,
    )
    np.testing.assert_almost_equal(
        automl.results["pipeline_results"][0]["validation_score"],
        0.0,
        decimal=4,
    )


def test_main_objective_problem_type_mismatch(X_y_binary):
    X, y = X_y_binary
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary", objective="R2")
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
            objective="MCC Binary",
        )
    with pytest.raises(ValueError, match="is not compatible with a"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            objective="MCC Multiclass",
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


def test_init_no_holdout_set(X_y_binary):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")
    assert automl.passed_holdout_set is False
    assert automl.X_holdout is None
    assert automl.y_holdout is None


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
        X_train=X,
        y_train=y,
        problem_type="regression",
        objective="R2",
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
    dummy_binary_pipeline,
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
        automl.add_to_rankings(dummy_binary_pipeline)
        assert automl.best_pipeline.name == dummy_binary_pipeline.name
        assert automl.best_pipeline.parameters == dummy_binary_pipeline.parameters
        assert (
            automl.best_pipeline.component_graph
            == dummy_binary_pipeline.component_graph
        )

        assert len(automl.rankings) == 2
        assert len(automl.full_rankings) == 2
        assert 0.1234 in automl.rankings["mean_cv_score"].values

    with env.test_context(score_return_value={"Log Loss Binary": 0.5678}):
        test_pipeline_2 = dummy_binary_pipeline.new(
            parameters={"Mock Classifier": {"a": 1.234}},
        )
        automl.add_to_rankings(test_pipeline_2)
        assert automl.best_pipeline.name == dummy_binary_pipeline.name
        assert automl.best_pipeline.parameters == dummy_binary_pipeline.parameters
        assert (
            automl.best_pipeline.component_graph
            == dummy_binary_pipeline.component_graph
        )
        assert len(automl.rankings) == 2
        assert len(automl.full_rankings) == 3
        assert 0.5678 not in automl.rankings["mean_cv_score"].values
        assert 0.5678 in automl.full_rankings["mean_cv_score"].values


def test_add_to_rankings_no_search(
    AutoMLTestEnv,
    dummy_binary_pipeline,
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
        automl.add_to_rankings(dummy_binary_pipeline)
        best_pipeline = automl.best_pipeline
        assert best_pipeline is not None
        assert isinstance(automl.data_splitter, StratifiedKFold)
        assert len(automl.rankings) == 1
        assert 0.5234 in automl.rankings["mean_cv_score"].values
        assert 0.5234 in automl.rankings["validation_score"].values
        assert np.isnan(
            automl.results["pipeline_results"][0]["percent_better_than_baseline"],
        )
        assert all(
            np.isnan(res)
            for res in automl.results["pipeline_results"][0][
                "percent_better_than_baseline_all_objectives"
            ].values()
        )


def test_add_to_rankings_regression_large(
    AutoMLTestEnv,
    dummy_regression_pipeline,
    example_regression_graph,
):
    X = pd.DataFrame(
        {"col_0": range(111113)},
    )  # Reaches just over max row threshold after holdout set
    y = pd.Series(range(111113))

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
        automl.add_to_rankings(dummy_regression_pipeline)
        assert isinstance(automl.data_splitter, TrainingValidationSplit)
        assert len(automl.rankings) == 1
        assert np.isnan(automl.rankings["mean_cv_score"].values[0])


def test_add_to_rankings_new_pipeline(dummy_regression_pipeline):
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
    automl.add_to_rankings(dummy_regression_pipeline)


def test_add_to_rankings_regression(
    example_regression_graph,
    dummy_regression_pipeline,
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
        automl.add_to_rankings(dummy_regression_pipeline)

    assert isinstance(automl.data_splitter, KFold)
    assert len(automl.rankings) == 1
    assert 0.1234 in automl.rankings["mean_cv_score"].values


def test_add_to_rankings_duplicate(
    AutoMLTestEnv,
    dummy_binary_pipeline,
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
    assert automl.best_pipeline == best_pipeline
    automl.add_to_rankings(dummy_binary_pipeline)

    test_pipeline_duplicate = dummy_binary_pipeline.new({})
    assert automl.add_to_rankings(test_pipeline_duplicate) is None


def test_add_to_rankings_trained(
    dummy_classifier_estimator_class,
    AutoMLTestEnv,
    dummy_binary_pipeline,
    X_y_binary,
):
    X, y = X_y_binary
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
        automl.add_to_rankings(dummy_binary_pipeline)
        assert len(automl.rankings) == 2
        assert len(automl.full_rankings) == 2
        assert list(automl.rankings["mean_cv_score"].values).count(0.1234) == 1
        assert list(automl.full_rankings["mean_cv_score"].values).count(0.1234) == 1

    with env.test_context(
        score_return_value={"Log Loss Binary": 0.1234},
        mock_fit_return_value=BinaryClassificationPipeline(
            component_graph=dummy_binary_pipeline.component_graph,
            parameters={},
        ),
    ):
        test_pipeline_trained = BinaryClassificationPipeline(
            component_graph=dummy_binary_pipeline.component_graph,
            parameters={},
        ).fit(X, y)
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
        "validation_score",
        "mean_cv_score",
        "standard_deviation_cv_score",
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
        PipelineNotFoundError,
        match="Pipeline not found in automl results",
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
        assert automl_dict["pipeline_summary"] == "Baseline Classifier w/ Label Encoder"
        assert automl_dict["parameters"] == {
            "Label Encoder": {"positive_label": None},
            "Baseline Classifier": {"strategy": "mode"},
        }
        assert automl_dict["mean_cv_score"] == 1.0
        assert not automl_dict["high_variance_cv"]
        assert isinstance(automl_dict["training_time"], float)
        assert automl_dict["cv_data"] == [
            {
                "all_objective_scores": OrderedDict(
                    [
                        ("Log Loss Binary", 1.0),
                        ("# Training", 66),
                        ("# Validation", 34),
                    ],
                ),
                "mean_cv_score": 1.0,
                "binary_classification_threshold": None,
            },
            {
                "all_objective_scores": OrderedDict(
                    [
                        ("Log Loss Binary", 1.0),
                        ("# Training", 67),
                        ("# Validation", 33),
                    ],
                ),
                "mean_cv_score": 1.0,
                "binary_classification_threshold": None,
            },
            {
                "all_objective_scores": OrderedDict(
                    [
                        ("Log Loss Binary", 1.0),
                        ("# Training", 67),
                        ("# Validation", 33),
                    ],
                ),
                "mean_cv_score": 1.0,
                "binary_classification_threshold": None,
            },
        ]
        assert automl_dict["percent_better_than_baseline_all_objectives"] == {
            "Log Loss Binary": 0,
        }
        assert automl_dict["percent_better_than_baseline"] == 0
        assert automl_dict["validation_score"] == 1.0
    else:
        assert automl_dict is None


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
    breast_cancer_local,
    wine_local,
    data_type,
    automl_type,
    target_type,
    make_data_type,
):
    if data_type == "np" and target_type in ["Int64", "boolean"]:
        pytest.skip(
            "Skipping test where data type is numpy and target type is nullable dtype",
        )

    if automl_type == ProblemTypes.BINARY:
        X, y = breast_cancer_local
        if "bool" in target_type:
            y = y.map({"malignant": False, "benign": True})
    elif automl_type == ProblemTypes.MULTICLASS:
        if "bool" in target_type:
            pytest.skip(
                "Skipping test where problem type is multiclass but target type is boolean",
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
        k=when_to_interrupt,
        starting_index=1,
    )
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=5,
        objective="f1",
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
        k=when_to_interrupt,
        starting_index=2,
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
        automl_algorithm="iterative",
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
            "validation_score": scores,
            "pipeline_name": [f"Mock name {i}" for i in range(len(scores))],
        },
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
    dummy_binary_pipeline,
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
            dummy_binary_pipeline,
            dummy_binary_pipeline,
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
    assert len(automl.errors) == 0
    with pytest.raises(
        AutoMLSearchException,
        match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective",
    ):
        with env.test_context(score_return_value={"Log Loss Binary": None}):
            automl.search()
    assert len(automl.errors) > 0
    for pipeline_name, pipeline_error in automl.errors.items():
        assert "Label Encoder" in pipeline_error["Parameters"]
        assert isinstance(pipeline_error["Exception"], TypeError)
        assert "line" in pipeline_error["Traceback"]


@patch("evalml.automl.automl_algorithm.IterativeAlgorithm.next_batch")
@patch("evalml.automl.AutoMLSearch.full_rankings", new_callable=PropertyMock)
@patch("evalml.automl.AutoMLSearch.rankings", new_callable=PropertyMock)
def test_pipelines_in_batch_return_none(
    mock_rankings,
    mock_full_rankings,
    mock_next_batch,
    X_y_binary,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline,
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
            dummy_binary_pipeline,
            dummy_binary_pipeline,
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
    assert len(automl.errors) == 0
    with pytest.raises(
        AutoMLSearchException,
        match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective",
    ):
        with env.test_context(score_return_value={"Log Loss Binary": None}):
            automl.search()
    assert len(automl.errors) > 0


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
    assert len(automl.errors) == 0
    with pytest.raises(
        AutoMLSearchException,
        match="All pipelines in the current AutoML batch produced a score of np.nan on the primary objective",
    ):
        with env.test_context(score_return_value={"Log Loss Binary": 1.0}):
            automl.search()
    for pipeline in automl.results["pipeline_results"].values():
        assert np.isnan(pipeline["mean_cv_score"])
    assert len(automl.errors) > 0


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
    dummy_classifier_estimator_class,
    dummy_regressor_estimator_class,
    dummy_time_series_regressor_estimator_class,
    ts_data,
    X_y_multi,
):
    if not objective.is_defined_for_problem_type(problem_type_value):
        pytest.skip("Skipping because objective is not defined for problem type")

    X, _, y = ts_data(problem_type=problem_type_value)

    estimator = {
        ProblemTypes.BINARY: dummy_classifier_estimator_class,
        ProblemTypes.MULTICLASS: dummy_classifier_estimator_class,
        ProblemTypes.REGRESSION: dummy_regressor_estimator_class,
        ProblemTypes.TIME_SERIES_REGRESSION: dummy_time_series_regressor_estimator_class,
    }[problem_type_value]
    baseline_pipeline_class = {
        ProblemTypes.BINARY: "evalml.pipelines.BinaryClassificationPipeline",
        ProblemTypes.MULTICLASS: "evalml.pipelines.MulticlassClassificationPipeline",
        ProblemTypes.REGRESSION: "evalml.pipelines.RegressionPipeline",
        ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesRegressionPipeline",
    }[problem_type_value]
    pipeline_class = _get_pipeline_base_class(problem_type_value)

    class DummyPipeline(pipeline_class):
        problem_type = problem_type_value

        def __init__(self, parameters, custom_name=None, random_seed=0):
            super().__init__(
                component_graph=[estimator],
                parameters=parameters,
                custom_name=custom_name,
            )

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
                "time_index": "date",
                "gap": 0,
                "max_delay": 0,
                "forecast_horizon": 2,
            },
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
            problem_configuration=pipeline_parameters["pipeline"],
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
    automl.automl_algorithm = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type=problem_type_value,
        max_iterations=2,
        tuner_class=SKOptTuner,
        random_seed=0,
        n_jobs=1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        search_parameters=pipeline_parameters,
    )
    automl.automl_algorithm._set_allowed_pipelines(allowed_pipelines)
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
            ),
        )
        baseline_name = next(
            name
            for name in automl.rankings.pipeline_name
            if name not in {"Pipeline1", "Pipeline2"}
        )
        answers = {
            "Pipeline1": round(
                objective.calculate_percent_difference(
                    pipeline_scores[0],
                    baseline_score,
                ),
                2,
            ),
            "Pipeline2": round(
                objective.calculate_percent_difference(
                    pipeline_scores[1],
                    baseline_score,
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
    "problem_type",
    ["binary", "multiclass", "regression", "time series regression"],
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
    dummy_classifier_estimator_class,
    dummy_regressor_estimator_class,
    dummy_time_series_regressor_estimator_class,
    ts_data,
):
    X, _, y = ts_data(problem_type=problem_type)

    problem_type_enum = handle_problem_types(problem_type)

    estimator = {
        "binary": dummy_classifier_estimator_class,
        "multiclass": dummy_classifier_estimator_class,
        "regression": dummy_regressor_estimator_class,
        "time series regression": dummy_time_series_regressor_estimator_class,
    }[problem_type]
    baseline_pipeline_class = {
        ProblemTypes.BINARY: "evalml.pipelines.BinaryClassificationPipeline",
        ProblemTypes.MULTICLASS: "evalml.pipelines.MulticlassClassificationPipeline",
        ProblemTypes.REGRESSION: "evalml.pipelines.RegressionPipeline",
        ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesRegressionPipeline",
    }[problem_type_enum]
    pipeline_class = _get_pipeline_base_class(problem_type_enum)

    class DummyPipeline(pipeline_class):
        name = "Dummy 1"
        problem_type = problem_type_enum

        def __init__(self, parameters, random_seed=0):
            super().__init__(component_graph=[estimator], parameters=parameters)

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
            get_default_primary_search_objective(problem_type_enum),
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
            mock_scores[obj_class.name],
            mock_baseline_scores[obj_class.name],
        )
        baseline_percent_difference[obj_class.name] = 0

    mock_score_1 = MagicMock(return_value=mock_scores)
    DummyPipeline.score = mock_score_1
    parameters = {}
    if problem_type_enum == ProblemTypes.TIME_SERIES_REGRESSION:
        parameters = {
            "pipeline": {
                "time_index": "date",
                "gap": 6,
                "max_delay": 3,
                "forecast_horizon": 3,
            },
        }
    # specifying problem_configuration for all problem types for conciseness
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        max_iterations=2,
        objective="auto",
        problem_configuration={
            "time_index": "date",
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 3,
        },
        optimize_thresholds=False,
        additional_objectives=additional_objectives,
    )
    automl.automl_algorithm = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type=problem_type,
        tuner_class=SKOptTuner,
        random_seed=0,
        n_jobs=-1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        search_parameters={
            "pipeline": {
                "time_index": "date",
                "gap": 1,
                "max_delay": 1,
                "forecast_horizon": 2,
            },
        },
    )
    automl.automl_algorithm._set_allowed_pipelines([DummyPipeline(parameters)])
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
    X, _, y = ts_data()
    X.index.name = "date"
    problem_configuration = {
        "time_index": "date",
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
    assert (
        automl.automl_algorithm.search_parameters["pipeline"] == problem_configuration
    )


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
    dummy_classifier_estimator_class,
    X_y_binary,
    AutoMLTestEnv,
):
    # Test that percent-better-than-baseline is correctly computed when scores differ across folds
    X, y = X_y_binary

    class DummyPipeline(BinaryClassificationPipeline):
        name = "Dummy 1"
        problem_type = ProblemTypes.BINARY

        def __init__(self, parameters, random_seed=0):
            super().__init__([dummy_classifier_estimator_class], parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    mock_score = MagicMock(
        side_effect=[{"Log Loss Binary": 1, "F1": val} for val in fold_scores],
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
    automl.automl_algorithm = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        max_iterations=2,
        tuner_class=SKOptTuner,
        random_seed=0,
        n_jobs=-1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        search_parameters={},
    )
    automl.automl_algorithm._set_allowed_pipelines([DummyPipeline({})])

    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 1, "F1": 1}):
        automl.search()
    assert (
        len(automl.results["pipeline_results"]) == 2
    ), "This tests assumes only one non-baseline pipeline was run!"
    pipeline_results = automl.results["pipeline_results"][1]
    np.testing.assert_equal(
        pipeline_results["percent_better_than_baseline_all_objectives"]["F1"],
        answer,
    )


@pytest.mark.parametrize("max_batches", [1, 5, 10])
@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.REGRESSION])
@pytest.mark.parametrize("automl_algorithm", ["iterative", "default"])
def test_max_batches_works(
    max_batches,
    problem_type,
    automl_algorithm,
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
        automl_algorithm=automl_algorithm,
    )
    automl.max_iterations = None
    env = AutoMLTestEnv(problem_type)
    with env.test_context(score_return_value={automl.objective.name: 0.3}):
        automl.search()
    assert automl._get_batch_number() == max_batches


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
    verbose,
    caplog,
    AutoMLTestEnv,
    logistic_regression_binary_pipeline,
    X_y_binary,
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
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"AUC": 0.5}):
        automl.search()
    assert not automl.progress.should_continue(automl._results)
    out = caplog.text
    assert (
        "2 iterations without improvement. Stopping search early." in out
    ) == verbose


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
    assert len(automl.results["pipeline_results"]) > 0

    # Use max_iterations when both max_iterations and max_batches
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

    # Respect max_batches when max_iterations is not set and algorithm is DefaultAlgorithm
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=1,
        optimize_thresholds=False,
        automl_algorithm="default",
    )
    assert automl.max_batches == 1
    assert automl.max_iterations is None

    with env.test_context(score_return_value={"Log Loss Binary": 0.3}):
        automl.search()

    assert len(automl.results["pipeline_results"]) == 3


@pytest.mark.parametrize("max_batches", [-1, -10, -np.inf])
def test_max_batches_must_be_non_negative(max_batches, X_y_binary):
    X, y = X_y_binary
    with pytest.raises(
        ValueError,
        match=f"Parameter max_batches must be None or non-negative. Received {max_batches}.",
    ):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            max_batches=max_batches,
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
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        n_jobs=1,
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
        },
    )
    X.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    y = [0, 1, 1, 0, 1, 0]
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.30}):
        automl.search()
    assert automl.rankings["pipeline_name"][1:-1].str.contains("Natural Language").all()


@pytest.mark.parametrize(
    "callback",
    [log_error_callback, silent_error_callback, raise_error_callback],
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
    else:
        assert len(automl.errors) > 0
        assert msg in caplog.text
        if callback == log_error_callback:
            assert f"Exception during automl search: {msg}" in caplog.text
        if callback in [raise_error_callback]:
            assert f"AutoML search raised a fatal exception: {msg}" in caplog.text


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
                arg.ww.logical_types["cat col"],
                ww.logical_types.Categorical,
            )
            assert arg.ww.semantic_tags["num col"] == {"numeric"}
            assert isinstance(arg.ww.logical_types["num col"], ww.logical_types.Integer)
            assert arg.ww.semantic_tags["text col"] == set()
            assert isinstance(
                arg.ww.logical_types["text col"],
                ww.logical_types.NaturalLanguage,
            )
    for arg in env.mock_score.call_args[0]:
        assert isinstance(arg, (pd.DataFrame, pd.Series))
        if isinstance(arg, pd.DataFrame):
            assert arg.ww.semantic_tags["cat col"] == {"category"}
            assert isinstance(
                arg.ww.logical_types["cat col"],
                ww.logical_types.Categorical,
            )
            assert arg.ww.semantic_tags["num col"] == {"numeric"}
            assert isinstance(arg.ww.logical_types["num col"], ww.logical_types.Integer)
            assert arg.ww.semantic_tags["text col"] == set()
            assert isinstance(
                arg.ww.logical_types["text col"],
                ww.logical_types.NaturalLanguage,
            )


def test_automl_validates_problem_configuration(ts_data):
    X, _, y = ts_data()
    assert (
        AutoMLSearch(X_train=X, y_train=y, problem_type="binary").problem_configuration
        == {}
    )
    assert (
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="multiclass",
        ).problem_configuration
        == {}
    )
    assert (
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
        ).problem_configuration
        == {}
    )
    msg = "problem_configuration must be a dict containing values for at least the time_index, gap, max_delay, and forecast_horizon parameters"
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

    with pytest.raises(ValueError, match="time_index cannot be None."):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="time series regression",
            problem_configuration={
                "time_index": None,
                "max_delay": 2,
                "gap": 3,
                "forecast_horizon": 2,
            },
        )

    problem_config = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="time series regression",
        problem_configuration={
            "time_index": "date",
            "max_delay": 2,
            "gap": 3,
            "forecast_horizon": 2,
        },
    ).problem_configuration
    assert problem_config == {
        "time_index": "date",
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
            [[set(train), set(test)] for train, test in a.data_splitter.split(X, y)],
        )
    # append split from last random state again, should be referencing same datasplit object
    data_splitters.append(
        [[set(train), set(test)] for train, test in a.data_splitter.split(X, y)],
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


def test_timeseries_baseline_init_with_correct_gap_max_delay(AutoMLTestEnv, ts_data):

    X, _, y = ts_data()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="time series regression",
        problem_configuration={
            "time_index": "date",
            "gap": 2,
            "max_delay": 3,
            "forecast_horizon": 1,
        },
        max_iterations=1,
    )
    env = AutoMLTestEnv("time series regression")
    with env.test_context():
        automl.search()

    # Best pipeline is baseline pipeline because we only run one iteration
    assert automl.best_pipeline.parameters == {
        "pipeline": {
            "time_index": "date",
            "gap": 2,
            "max_delay": 0,
            "forecast_horizon": 1,
        },
        "Time Series Featurizer": {
            "time_index": "date",
            "delay_features": False,
            "delay_target": True,
            "max_delay": 0,
            "gap": 2,
            "forecast_horizon": 1,
            "conf_level": 0.05,
            "rolling_window_size": 0.25,
        },
        "Time Series Baseline Estimator": {"forecast_horizon": 1, "gap": 2},
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
    problem_type,
    X_y_regression,
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
            "time_index": 0,
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
        search_parameters=params,
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
            "numeric_impute_strategy": Categorical(["median", "most_frequent"]),
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
        search_parameters=hyperparams,
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
                random_state=automl.random_seed,
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
        search_parameters=hyperparams,
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
    "ranges",
    [0, [float("-inf"), float("inf")], [float("-inf"), 0], [0, float("inf")]],
)
def test_automl_check_for_high_variance(ranges, X_y_binary, dummy_binary_pipeline):
    X, y = X_y_binary
    if ranges == 0:
        objectives = "Log Loss Binary"
    else:
        objectives = CustomClassificationObjectiveRanges(ranges)
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective=objectives,
    )
    cv_scores = pd.Series([1, 1, 1])
    pipeline = dummy_binary_pipeline
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
    dummy_classifier_estimator_class,
    X_y_binary,
):
    X, y = X_y_binary

    pipeline_0 = BinaryClassificationPipeline(
        custom_name="Custom Pipeline",
        component_graph=[dummy_classifier_estimator_class],
    )
    pipeline_1 = BinaryClassificationPipeline(
        custom_name="Custom Pipeline",
        component_graph=[dummy_classifier_estimator_class],
    )
    pipeline_2 = BinaryClassificationPipeline(
        custom_name="My Pipeline 3",
        component_graph=[dummy_classifier_estimator_class],
    )
    pipeline_3 = BinaryClassificationPipeline(
        custom_name="My Pipeline 3",
        component_graph=[dummy_classifier_estimator_class],
    )

    with pytest.raises(
        ValueError,
        match="All pipeline names must be unique. The name 'Custom Pipeline' was repeated.",
    ):
        AutoMLSearch(X, y, problem_type="binary").train_pipelines(
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
        AutoMLSearch(X, y, problem_type="binary").train_pipelines(
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
        AutoMLSearch(X, y, problem_type="binary").score_pipelines(
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
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    X_y_binary,
):
    custom_names = [f"Pipeline {i}" for i in range(3)]
    pipelines = [
        BinaryClassificationPipeline(
            component_graph=[dummy_classifier_estimator_class],
            custom_name=custom_name,
        )
        for custom_name in custom_names
    ]

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
    dummy_classifier_estimator_class,
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

    pipelines = [
        BinaryClassificationPipeline(
            component_graph=[dummy_classifier_estimator_class],
            custom_name=f"Pipeline {index}",
            parameters={"Mock Classifier": {"a": index}},
        )
        for index in range(len(pipeline_fit_side_effect) - 1)
    ]
    input_pipelines = [
        BinaryClassificationPipeline([classifier])
        for classifier in stackable_classifiers[:2]
    ]
    ensemble = BinaryClassificationPipeline(
        [StackedEnsembleClassifier],
        parameters={
            "Stacked Ensemble Classifier": {
                "input_pipelines": input_pipelines,
                "n_jobs": 1,
            },
        },
    )
    pipelines.append(ensemble)
    env = AutoMLTestEnv("binary")

    def train_batch_and_check():
        caplog.clear()
        with env.test_context(mock_fit_side_effect=pipeline_fit_side_effect):
            trained_pipelines = automl.train_pipelines(pipelines)

            assert len(trained_pipelines) == len(pipeline_fit_side_effect) - len(
                exceptions_to_check,
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
    stackable_classifiers,
    caplog,
):

    exceptions_to_check = []
    expected_scores = {}
    for i, e in enumerate(pipeline_score_side_effect):
        # Ensemble pipeline has different name
        pipeline_name = f"Pipeline {i}"
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
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
        optimize_thresholds=False,
    )
    env = AutoMLTestEnv("binary")

    pipelines = [
        BinaryClassificationPipeline(
            component_graph=[dummy_classifier_estimator_class],
            custom_name=f"Pipeline {index}",
            parameters={"Mock Classifier": {"a": index}},
        )
        for index in range(len(pipeline_score_side_effect) - 1)
    ]
    input_pipelines = [
        BinaryClassificationPipeline([classifier])
        for classifier in stackable_classifiers[:2]
    ]
    ensemble = _make_stacked_ensemble_pipeline(input_pipelines, ProblemTypes.BINARY)
    ensemble._custom_name = f"Pipeline {len(pipeline_score_side_effect) - 1}"
    pipelines.append(ensemble)

    def score_batch_and_check():
        caplog.clear()
        with env.test_context(mock_score_side_effect=pipeline_score_side_effect):

            scores = automl.score_pipelines(
                pipelines,
                X,
                y,
                objectives=["Log Loss Binary", "F1", "AUC"],
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
    X_y_binary,
    dummy_classifier_estimator_class,
):
    X, y = X_y_binary
    pipeline_1 = BinaryClassificationPipeline(
        component_graph=[dummy_classifier_estimator_class],
        custom_name="Mock Pipeline",
    )
    pipeline_2 = BinaryClassificationPipeline(
        component_graph=[dummy_classifier_estimator_class],
        custom_name="Mock Pipeline",
    )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
    )

    with pytest.raises(
        ValueError,
        match="All pipeline names must be unique. The name 'Mock Pipeline' was repeated.",
    ):
        automl.train_pipelines([pipeline_1, pipeline_2])

    with pytest.raises(
        ValueError,
        match="All pipeline names must be unique. The name 'Mock Pipeline' was repeated.",
    ):
        automl.score_pipelines([pipeline_1, pipeline_2], X, y, None)


def test_score_batch_before_fitting_yields_error_nan_scores(
    X_y_binary,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline,
    caplog,
):
    X, y = X_y_binary

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_iterations=1,
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
    )

    scored_pipelines = automl.score_pipelines(
        [dummy_binary_pipeline],
        X,
        y,
        objectives=["Log Loss Binary", F1()],
    )
    assert scored_pipelines == {
        "Mock Binary Classification Pipeline": {
            "Log Loss Binary": np.nan,
            "F1": np.nan,
        },
    }

    assert "Score error for Mock Binary Classification Pipeline" in caplog.text


def test_high_cv_check_no_warning_for_divide_by_zero(X_y_binary, dummy_binary_pipeline):
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary")
    with pytest.warns(None) as warnings:
        # mean is 0 but std is not
        automl._check_for_high_variance(
            dummy_binary_pipeline,
            cv_scores=[0.0, 1.0, -1.0],
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
    dummy_binary_pipeline,
    dummy_multiclass_pipeline,
    AutoMLTestEnv,
):
    if automl_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        y = pd.Series(y).map({0: -5.19, 1: 6.7})
        mock_train.return_value = dummy_binary_pipeline
    else:
        X, y = X_y_multi
        y = pd.Series(y).map({0: -5.19, 1: 6.7, 2: 2.03})
        mock_train.return_value = dummy_multiclass_pipeline

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
                "time_index": 0,
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
            "columns": ["index_col"],
        }

    all_drop_column_params = []
    for _, row in automl.full_rankings.iterrows():
        if "Baseline" not in row.pipeline_name:
            all_drop_column_params.append(
                row.parameters["Drop Columns Transformer"]["columns"],
            )
    assert all(param == ["index_col"] for param in all_drop_column_params)


def test_automl_validates_data_passed_in_to_allowed_component_graphs(
    X_y_binary,
    dummy_classifier_estimator_class,
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
                        dummy_classifier_estimator_class,
                    ],
                },
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
                "Mock Binary Classification Pipeline": dummy_classifier_estimator_class,
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
            {10: [1.0, 1.0, 1.0, 1.0], 11: [0.0, 0.0, 0.0, 0.0]},
        )
    if problem_type == ProblemTypes.MULTICLASS:
        expected_predictions = pd.Series(np.array([11] * len(X)), dtype="int64")
        expected_predictions_proba = pd.DataFrame(
            {
                10: [0.0, 0.0, 0.0, 0.0],
                11: [1.0, 1.0, 1.0, 1.0],
                12: [0.0, 0.0, 0.0, 0.0],
            },
        )
    if problem_type == ProblemTypes.REGRESSION:
        mean = y.mean()
        expected_predictions = pd.Series([mean] * len(X))

    pd.testing.assert_series_equal(expected_predictions, baseline.predict(X))
    if is_classification(problem_type):
        pd.testing.assert_frame_equal(
            expected_predictions_proba,
            baseline.predict_proba(X),
        )
    np.testing.assert_allclose(
        baseline.feature_importance.iloc[:, 1],
        np.array([0.0] * X.shape[1]),
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
    X = pd.DataFrame(
        {"a": [4, 5, 6, 7, 8], "b": pd.date_range("2021-01-01", periods=5)},
    )
    y = pd.Series([0, 1, 1, 0, 1])
    expected_predictions_proba = pd.DataFrame(
        {
            0: pd.Series([1.0], index=[4]),
            1: pd.Series([0.0], index=[4]),
        },
    )
    if problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        y = pd.Series([0, 2, 0, 1, 1])
        expected_predictions_proba = pd.DataFrame(
            {
                0: pd.Series([0.0], index=[4]),
                1: pd.Series([1.0], index=[4]),
                2: pd.Series([0.0], index=[4]),
            },
        )

    automl = AutoMLSearch(
        X,
        y,
        problem_type=problem_type,
        problem_configuration={
            "time_index": "b",
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
    expected_predictions = expected_predictions
    if problem_type != ProblemTypes.TIME_SERIES_REGRESSION:
        expected_predictions = pd.Series(
            expected_predictions,
            name="target_delay_1",
            dtype="int64",
        )

    preds = baseline.predict(X_validation, None, X_train, y_train)
    pd.testing.assert_series_equal(expected_predictions, preds)
    if is_classification(problem_type):
        pd.testing.assert_frame_equal(
            expected_predictions_proba,
            baseline.predict_proba(X_validation, X_train, y_train),
        )
    transformed = baseline.transform_all_but_final(X_train, y_train)
    importance = np.array([0] * transformed.shape[1])
    importance[0] = 1
    np.testing.assert_allclose(baseline.feature_importance.iloc[:, 1], importance)


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
    for col in columns:
        X.ww[col] = pd.Series(range(len(X)))
    X.ww.set_types({col: "Unknown" for col in columns})
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_batches=3,
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
                row.parameters["Drop Columns Transformer"]["columns"],
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
    automl_type,
    AutoMLTestEnv,
    X_y_binary,
    X_y_multi,
    X_y_regression,
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
            "time_index": 0,
            "forecast_horizon": 10,
        }
        X, y = X_y_regression
        X.index = pd.DatetimeIndex(pd.date_range("01-01-2022", periods=len(X)))
    else:
        problem_configuration = {
            "gap": 1,
            "max_delay": 1,
            "time_index": 0,
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


@patch("evalml.pipelines.utils._get_preprocessing_components")
@pytest.mark.parametrize("verbose", [True, False])
def test_component_and_pipeline_warnings_surface_in_search(
    mock_get_preprocessing_components,
    verbose,
    AutoMLTestEnv,
    X_y_regression,
):
    X, y = X_y_regression

    def dummy_mock_get_preprocessing_components(*args, **kwargs):
        warnings.warn(UserWarning("dummy test warning"))
        return ["Imputer"]

    mock_get_preprocessing_components.side_effect = (
        dummy_mock_get_preprocessing_components
    )
    with pytest.warns(None) as warnings_logged:
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
            search_parameters={"Decision Tree Classifier": {"max_depth": 1}},
            max_batches=1,
            verbose=verbose,
        )
        env = AutoMLTestEnv("binary")
        with env.test_context(score_return_value={automl.objective.name: 1.0}):
            automl.search()

    found_user = False
    found_parameter = False
    for warning in warnings_logged:
        if isinstance(warning.message, UserWarning):
            found_user = True
        if isinstance(warning.message, ParameterNotUsedWarning):
            found_parameter = True
        if found_user and found_parameter:
            continue

    assert len(warnings_logged)
    assert found_user and found_parameter


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
        },
    )
    X.ww.init(logical_types={"b": "NaturalLanguage"})
    y = pd.Series([0] * 25 + [1] * 75)
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_batches=1,
    )
    automl.search()
    for (x, _), _ in mock_fit.call_args_list:
        assert all(
            [str(types) == "Double" for types in x.ww.types["Logical Type"].values],
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
            ValueError,
            match="is not a valid engine, please choose from",
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
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=engine_choice,
        )
        assert isinstance(automl._engine, DaskEngine)
        automl.close_engine()
    elif engine_choice == "engine_instance":
        engine_choice = DaskEngine()
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=engine_choice,
        )
        automl.close_engine()
    elif engine_choice == "invalid_str":
        engine_choice = "DaskEngine"
        with pytest.raises(
            ValueError,
            match="is not a valid engine, please choose from",
        ):
            automl = AutoMLSearch(
                X_train=X,
                y_train=y,
                problem_type="binary",
                engine=engine_choice,
            )
    elif engine_choice == "invalid_type":
        engine_choice = DaskEngine
        with pytest.raises(
            TypeError,
            match="Invalid type provided for 'engine'.  Requires string, DaskEngine instance, or CFEngine instance.",
        ):
            automl = AutoMLSearch(
                X_train=X,
                y_train=y,
                problem_type="binary",
                engine=engine_choice,
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
        problem_type="regression",
        allowed_component_graphs=component_graphs,
        ensembling=True,
        max_batches=4,
        verbose=True,
    )
    automl.search()
    assert "Stacked Ensemble Regression Pipeline" in caplog.text
    assert "Stacked Ensemble Regression Pipeline" in list(
        automl.rankings["pipeline_name"],
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
            component_graph={
                "Label Encoder": ["Label Encoder", "X", "y"],
                "Baseline Classifier": [
                    "Baseline Classifier",
                    "Label Encoder.x",
                    "Label Encoder.y",
                ],
            },
            custom_name="Mode Baseline Binary Classification Pipeline",
            parameters={"Baseline Classifier": {"strategy": "mode"}},
        )
    elif automl_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        score_value = {"Log Loss Multiclass": 1.0}
        expected_pipeline = MulticlassClassificationPipeline(
            component_graph={
                "Label Encoder": ["Label Encoder", "X", "y"],
                "Baseline Classifier": [
                    "Baseline Classifier",
                    "Label Encoder.x",
                    "Label Encoder.y",
                ],
            },
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


@pytest.mark.parametrize(
    "problem_type",
    [
        "time series regression",
        "time series multiclass",
        "time series binary",
    ],
)
def test_automl_passes_known_in_advance_pipeline_parameters_to_all_pipelines(
    problem_type,
    ts_data,
    AutoMLTestEnv,
):
    X, _, y = ts_data(problem_type=problem_type)

    X.ww["email"] = pd.Series(["foo@foo.com"] * X.shape[0], index=X.index)
    X.ww["category"] = pd.Series(["a"] * X.shape[0], index=X.index)
    X.ww.set_types({"email": "EmailAddress", "category": "Categorical"})
    known_in_advance = ["email", "category"]

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        max_batches=3,
        problem_configuration={
            "time_index": "date",
            "max_delay": 3,
            "forecast_horizon": 3,
            "gap": 1,
            "known_in_advance": known_in_advance,
        },
    )

    test_env = AutoMLTestEnv(problem_type)
    with test_env.test_context(score_return_value={automl.objective.name: 0.02}):
        automl.search()

    no_baseline = automl.full_rankings.loc[
        ~automl.full_rankings.pipeline_name.str.contains("Baseline")
    ]
    assert no_baseline.parameters.map(
        lambda d: d["Known In Advance Pipeline - Select Columns Transformer"]["columns"]
        == known_in_advance,
    ).all()
    assert no_baseline.parameters.map(
        lambda d: d["Not Known In Advance Pipeline - Select Columns Transformer"][
            "columns"
        ]
        == ["feature", "date"],
    ).all()


@pytest.mark.parametrize(
    "data_splitter,mean_cv_is_none",
    [(TrainingValidationSplit, True), (StratifiedKFold, False)],
)
def test_cv_validation_scores(
    data_splitter,
    mean_cv_is_none,
    dummy_classifier_estimator_class,
    AutoMLTestEnv,
):
    X, y = datasets.make_classification(
        n_samples=500,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        random_state=0,
    )
    data_splitter = data_splitter()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=3,
        data_splitter=data_splitter,
        allowed_component_graphs={"Name": [dummy_classifier_estimator_class]},
        n_jobs=1,
    )
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.5}):
        automl.search()
    cv_vals = list(set(automl.full_rankings["mean_cv_score"].values))
    validation_vals = list(set(automl.full_rankings["validation_score"].values))
    assert len(validation_vals) == 1
    assert validation_vals[0] == 0.5
    if mean_cv_is_none:
        assert np.isnan(cv_vals[0])
    else:
        assert cv_vals[0] == validation_vals[0]


def test_cv_validation_scores_time_series(
    ts_data,
    AutoMLTestEnv,
):
    X, _, y = ts_data(problem_type="time series binary")
    problem_configuration = {
        "time_index": "date",
        "gap": 0,
        "max_delay": 0,
        "forecast_horizon": 2,
    }
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="time series binary",
        max_iterations=3,
        data_splitter=TimeSeriesSplit(n_splits=3),
        problem_configuration=problem_configuration,
        n_jobs=1,
    )
    env = AutoMLTestEnv("time series binary")
    with env.test_context(score_return_value={"Log Loss Binary": 0.5}):
        automl.search()
    cv_vals = list(set(automl.full_rankings["mean_cv_score"].values))
    validation_vals = list(set(automl.full_rankings["validation_score"].values))
    assert len(validation_vals) == 1
    assert validation_vals[0] == 0.5
    assert cv_vals[0] == validation_vals[0]


@pytest.mark.parametrize("algorithm,batches", [("iterative", 2), ("default", 3)])
@pytest.mark.parametrize(
    "parameter,expected",
    [
        ("mean", ["mean", "median", "most_frequent", "knn"]),
        (Categorical(["mean"]), Categorical(["mean"])),
    ],
)
@pytest.mark.parametrize("problem_type", ["binary", "time series binary"])
def test_search_parameters_held_automl(
    problem_type,
    parameter,
    expected,
    algorithm,
    batches,
    X_y_binary,
    ts_data,
):
    if problem_type == "binary":
        X, y = X_y_binary
        problem_configuration = None
        allowed_component_graphs = {
            "cg": {
                "Imputer": ["Imputer", "X", "y"],
                "Label Encoder": ["Label Encoder", "Imputer.x", "y"],
                "Decision Tree Classifier": [
                    "Decision Tree Classifier",
                    "Label Encoder.x",
                    "Label Encoder.y",
                ],
            },
        }
    else:
        X, _, y = ts_data(problem_type="time series binary")
        problem_configuration = {
            "time_index": "date",
            "gap": 0,
            "max_delay": 0,
            "forecast_horizon": 3,
        }
        allowed_component_graphs = {
            "cg": {
                "Imputer": ["Imputer", "X", "y"],
                "Label Encoder": ["Label Encoder", "Imputer.x", "y"],
                "DateTime Featurizer": [
                    "DateTime Featurizer",
                    "Label Encoder.x",
                    "Label Encoder.y",
                ],
                "Decision Tree Classifier": [
                    "Decision Tree Classifier",
                    "DateTime Featurizer.x",
                    "Label Encoder.y",
                ],
            },
        }
    search_parameters = {
        "Imputer": {"numeric_impute_strategy": parameter},
        "DateTime Featurizer": {"features_to_extract": ["month", "day_of_week"]},
        "Label Encoder": {"positive_label": 0},
    }
    aml = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        problem_configuration=problem_configuration,
        allowed_component_graphs=allowed_component_graphs,
        search_parameters=search_parameters,
        automl_algorithm=algorithm,
        max_batches=batches,
    )
    aml.search()
    estimator_args = inspect.getargspec(DecisionTreeClassifier)
    # estimator_args[0] gives the parameter names, while [3] gives the associated values
    # estimator_args[0][i + 1] to skip 'self' in the estimator
    # we do len - 1 in order to skip the random seed, which isn't present in the row['parameters']
    expected_params = {
        estimator_args[0][i + 1]: estimator_args[3][i]
        for i in range(len(estimator_args[3]) - 1)
    }
    sorted_full_rank = aml.full_rankings.sort_values(by="id")
    found_dtc = False
    for _, row in sorted_full_rank.iterrows():
        # we check the initial decision tree classifier parameters.
        if "Decision Tree Classifier" in row["parameters"]:
            assert expected_params == row["parameters"]["Decision Tree Classifier"]
            found_dtc = True
            break
    assert found_dtc
    for tuners in aml.automl_algorithm._tuners.values():
        assert (
            tuners._pipeline_hyperparameter_ranges["Imputer"]["numeric_impute_strategy"]
            == expected
        )
        assert tuners._pipeline_hyperparameter_ranges["Imputer"][
            "categorical_impute_strategy"
        ] == ["most_frequent"]
        # make sure that there are no set hyperparameters when we don't have defaults
        assert tuners._pipeline_hyperparameter_ranges["Label Encoder"] == {}
        assert tuners.propose()["Label Encoder"] == {}
        if problem_type == "time series binary":
            assert tuners._pipeline_hyperparameter_ranges["DateTime Featurizer"] == {}


@pytest.mark.parametrize(
    "automl_algorithm",
    ["iterative", "default"],
)
@pytest.mark.parametrize(
    "features",
    ["with_features_provided", "without_features_provided"],
)
def test_automl_accepts_features(
    automl_algorithm,
    features,
    X_y_binary,
    AutoMLTestEnv,
):
    X, y = X_y_binary
    X = pd.DataFrame(X)  # Drop ww information since setting column types fails
    X.columns = X.columns.astype(str)
    X_transform = X.iloc[len(X) // 3 :]

    if features == "with_features_provided":
        es = ft.EntitySet()
        es = es.add_dataframe(
            dataframe_name="X",
            dataframe=X_transform,
            index="index",
            make_index=True,
        )
        _, features = ft.dfs(
            entityset=es,
            target_dataframe_name="X",
            trans_primitives=["absolute"],
        )
    else:
        features = None

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        optimize_thresholds=False,
        max_batches=3,
        features=features,
        automl_algorithm=automl_algorithm,
    )

    assert automl.automl_algorithm.features == features
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    if features:
        assert all(
            [
                p["DFS Transformer"]["features"] == features
                for p in automl.full_rankings["parameters"][1:]
            ],
        )
    else:
        assert all(
            [
                "DFS Transformer" not in p
                for p in automl.full_rankings["parameters"][1:]
            ],
        )


@pytest.mark.skip_during_conda
def test_automl_with_iterative_algorithm_puts_ts_estimators_first(
    ts_data,
    AutoMLTestEnv,
    is_using_windows,
):

    X, _, y = ts_data()

    env = AutoMLTestEnv("time series regression")
    automl = AutoMLSearch(
        X,
        y,
        problem_type="time series regression",
        max_iterations=9,
        problem_configuration={
            "max_delay": 2,
            "gap": 0,
            "forecast_horizon": 2,
            "time_index": "Date",
        },
        verbose=True,
        automl_algorithm="iterative",
    )
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    estimator_order = (
        automl.full_rankings.sort_values("search_order")
        .id.map(lambda id_: automl.get_pipeline(id_).estimator.name)
        .tolist()
    )
    if is_using_windows:
        expected_order = [
            "Time Series Baseline Estimator",
            "ARIMA Regressor",
            "ARIMA Regressor",
            "Exponential Smoothing Regressor",
            "Exponential Smoothing Regressor",
            "Elastic Net Regressor",
            "Elastic Net Regressor",
            "XGBoost Regressor",
            "XGBoost Regressor",
        ]
    else:
        expected_order = [
            "Time Series Baseline Estimator",
            "ARIMA Regressor",
            "ARIMA Regressor",
            "Prophet Regressor",
            "Prophet Regressor",
            "Exponential Smoothing Regressor",
            "Exponential Smoothing Regressor",
            "Elastic Net Regressor",
            "Elastic Net Regressor",
        ]
    assert estimator_order == expected_order


@pytest.mark.skip_during_conda
@pytest.mark.parametrize("automl_algo", ["iterative", "default"])
@pytest.mark.parametrize(
    "hyperparams",
    [
        None,
        {"Imputer": {"numeric_impute_strategy": Categorical(["most_frequent"])}},
        {"ARIMA Regressor": {"seasonal": Categorical([True])}},
    ],
)
def test_automl_restricts_use_covariates_for_arima(
    hyperparams,
    automl_algo,
    AutoMLTestEnv,
    is_using_windows,
    X_y_binary,
):

    X, y = X_y_binary
    X = pd.DataFrame(X)
    X["Date"] = pd.date_range("2010-01-01", periods=X.shape[0])

    env = AutoMLTestEnv("time series regression")
    automl = AutoMLSearch(
        X,
        y,
        problem_type="time series regression",
        problem_configuration={
            "max_delay": 2,
            "gap": 0,
            "forecast_horizon": 2,
            "time_index": "Date",
        },
        verbose=True,
        search_parameters=hyperparams,
        automl_algorithm=automl_algo,
        max_batches=6,
    )
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    params = automl.full_rankings.parameters.map(
        lambda p: p.get("ARIMA Regressor", {}).get("use_covariates"),
    ).tolist()
    arima_params = [p for p in params if p is not None]
    assert arima_params
    assert all(not p for p in arima_params)


@pytest.mark.skip_during_conda
@pytest.mark.parametrize("automl_algo", ["iterative", "default"])
@pytest.mark.parametrize(
    "hyperparams",
    [
        {"ARIMA Regressor": {"use_covariates": Categorical([True])}},
        {
            "ARIMA Regressor": {"use_covariates": Categorical([True])},
            "Imputer": {"numeric_impute_strategy": Categorical(["most_frequent"])},
        },
    ],
)
def test_automl_does_not_restrict_use_covariates_if_user_specified(
    hyperparams,
    automl_algo,
    AutoMLTestEnv,
    is_using_windows,
    X_y_binary,
):

    X, y = X_y_binary
    X = pd.DataFrame(X)
    X["Date"] = pd.date_range("2010-01-01", periods=X.shape[0])
    env = AutoMLTestEnv("time series regression")
    automl = AutoMLSearch(
        X,
        y,
        problem_type="time series regression",
        problem_configuration={
            "max_delay": 2,
            "gap": 0,
            "forecast_horizon": 2,
            "time_index": "Date",
        },
        verbose=True,
        automl_algorithm=automl_algo,
        search_parameters=hyperparams,
        max_batches=6,
    )
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    params = automl.full_rankings.parameters.map(
        lambda p: p.get("ARIMA Regressor", {}).get("use_covariates"),
    ).tolist()
    arima_params = [p for p in params if p is not None]
    assert arima_params
    assert all(p for p in arima_params)


@pytest.mark.parametrize("automl_algo", ["iterative", "default"])
def test_automl_passes_down_ensembling(automl_algo, AutoMLTestEnv, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    env = AutoMLTestEnv("binary")
    max_batches = 4 if automl_algo == "default" else None
    max_iterations = (
        None if automl_algo == "default" else _get_first_stacked_classifier_no()
    )
    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        verbose=True,
        automl_algorithm=automl_algo,
        ensembling=True,
        max_batches=max_batches,
        max_iterations=max_iterations,
    )

    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    pipeline_names = automl.rankings["pipeline_name"]
    assert pipeline_names.str.contains("Ensemble").any()


def test_default_algorithm_uses_n_jobs(X_y_binary, AutoMLTestEnv):
    X, y = X_y_binary

    aml = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        max_batches=3,
        automl_algorithm="default",
        n_jobs=2,
    )

    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={aml.objective.name: 1.0}):
        aml.search()

    n_checked = 0
    n_feature_selector_checked = 0
    for pipeline_id in aml.rankings.id:
        pl = aml.get_pipeline(pipeline_id)
        if hasattr(pl.estimator._component_obj, "n_jobs"):
            n_checked += 1
            assert pl.estimator._component_obj.n_jobs == 2
        if "RF Classifier Select From Model" in pl.component_graph.component_instances:
            n_feature_selector_checked += 1
            assert (
                pl.get_component(
                    "RF Classifier Select From Model",
                )._component_obj.estimator.n_jobs
                == 2
            )

    assert n_checked and n_feature_selector_checked


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("automl_algorithm", ["default", "iterative"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_exclude_featurizers(
    automl_algorithm,
    problem_type,
    input_type,
    get_test_data_from_configuration,
    AutoMLTestEnv,
):
    parameters = {}
    if is_time_series(problem_type):
        parameters = {
            "time_index": "dates",
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 1,
        }

    X, y = get_test_data_from_configuration(
        input_type,
        problem_type,
        column_names=["dates", "text", "email", "url"],
    )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        problem_configuration=parameters,
        automl_algorithm=automl_algorithm,
        exclude_featurizers=[
            "DatetimeFeaturizer",
            "EmailFeaturizer",
            "URLFeaturizer",
            "NaturalLanguageFeaturizer",
            "TimeSeriesFeaturizer",
        ],
    )

    env = AutoMLTestEnv(problem_type)
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    pipelines = [
        automl.get_pipeline(i) for i in range(len(automl.results["pipeline_results"]))
    ]

    # A check to make sure we actually retrieve constructed pipelines from the algo.
    assert len(pipelines) > 0

    assert not any(
        [
            DateTimeFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )
    assert not any(
        [EmailFeaturizer.name in pl.component_graph.compute_order for pl in pipelines],
    )
    assert not any(
        [URLFeaturizer.name in pl.component_graph.compute_order for pl in pipelines],
    )
    assert not any(
        [
            NaturalLanguageFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )
    assert not any(
        [
            TimeSeriesFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )


def test_exclude_featurizers_errors(X_y_binary):
    X, y = X_y_binary
    match_text = (
        "Invalid value provided for exclude_featurizers. Must be one of: "
        "DatetimeFeaturizer, EmailFeaturizer, URLFeaturizer, NaturalLanguageFeaturizer, TimeSeriesFeaturizer"
    )
    with pytest.raises(
        ValueError,
        match=match_text,
    ):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            exclude_featurizers=[
                "InvalidNameFeaturizer",
            ],
        )

    problem_configuration = {
        "gap": 0,
        "max_delay": 7,
        "forecast_horizon": 7,
        "time_index": "date",
    }
    match_text = "For time series problems, if DatetimeFeaturizer is excluded, must also exclude TimeSeriesFeaturizer"
    with pytest.raises(
        ValueError,
        match=match_text,
    ):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="time series regression",
            problem_configuration=problem_configuration,
            exclude_featurizers=[
                "DatetimeFeaturizer",
            ],
        )

    match_text = "For time series problems, if TimeSeriesFeaturizer is excluded, must also exclude DatetimeFeaturizer"
    with pytest.raises(
        ValueError,
        match=match_text,
    ):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="time series multiclass",
            problem_configuration=problem_configuration,
            exclude_featurizers=[
                "TimeSeriesFeaturizer",
            ],
        )


def test_init_holdout_set(X_y_binary, caplog):
    X, y = X_y_binary
    X_train, X_holdout, y_train, y_holdout = split_data(X, y, "binary")

    match_text = "Must specify training data target values as a 1d vector using the y_holdout argument"
    with pytest.raises(
        ValueError,
        match=match_text,
    ):
        AutoMLSearch(
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_holdout,
            problem_type="binary",
        )
    match_text = "Must specify holdout data as a 2d array using the X_holdout argument"
    with pytest.raises(
        ValueError,
        match=match_text,
    ):
        AutoMLSearch(
            X_train=X_train,
            y_train=y_train,
            y_holdout=y_holdout,
            problem_type="binary",
        )

    automl = AutoMLSearch(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        problem_type="binary",
        verbose=True,
    )
    assert automl.passed_holdout_set is True
    assert (
        "AutoMLSearch will use the holdout set to score and rank pipelines."
        in caplog.text
    )
    assert_frame_equal(automl.X_holdout, X_holdout)
    assert_series_equal(automl.y_holdout, y_holdout)


def test_init_create_holdout_set(caplog):
    caplog.clear()
    X, y = datasets.make_classification(
        n_samples=AutoMLSearch._HOLDOUT_SET_MIN_ROWS - 1,
        random_state=0,
    )
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        verbose=True,
    )
    out = caplog.text

    match_text = f"Dataset size is too small to create holdout set. Minimum dataset size is {AutoMLSearch._HOLDOUT_SET_MIN_ROWS} rows, X_train has {len(X)} rows. Holdout set evaluation is disabled."
    assert match_text not in out

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        verbose=True,
        holdout_set_size=0.1,
    )
    out = caplog.text

    match_text = f"Dataset size is too small to create holdout set. Minimum dataset size is {AutoMLSearch._HOLDOUT_SET_MIN_ROWS} rows, X_train has {len(X)} rows. Holdout set evaluation is disabled."
    assert match_text in out
    assert "AutoMLSearch will use mean CV score to rank pipelines." in out
    assert len(automl.X_train) == len(X)
    assert len(automl.y_train) == len(y)
    assert automl.X_holdout is None
    assert automl.y_holdout is None
    assert automl.passed_holdout_set is False

    caplog.clear()
    X, y = datasets.make_classification(
        n_samples=AutoMLSearch._HOLDOUT_SET_MIN_ROWS,
        random_state=0,
    )
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        verbose=True,
        holdout_set_size=0.1,
    )
    out = caplog.text

    expected_holdout_size = int(automl.holdout_set_size * len(X))
    expected_train_size = int((1 - automl.holdout_set_size) * len(X))
    match_text = f"Created a holdout dataset with {expected_holdout_size} rows. Training dataset has {expected_train_size} rows."
    assert match_text in out
    assert "AutoMLSearch will use the holdout set to score and rank pipelines." in out
    assert len(automl.X_train) == expected_train_size
    assert len(automl.y_train) == expected_train_size
    assert len(automl.X_holdout) == expected_holdout_size
    assert len(automl.y_holdout) == expected_holdout_size
    assert automl.passed_holdout_set is False

    caplog.clear()
    X, y = datasets.make_classification(
        n_samples=AutoMLSearch._HOLDOUT_SET_MIN_ROWS,
        random_state=0,
    )
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        verbose=True,
        holdout_set_size=0,
    )
    out = caplog.text

    match_text = f"AutoMLSearch will use mean CV score to rank pipelines."
    assert match_text in out
    assert len(automl.X_train) == len(X)
    assert len(automl.y_train) == len(y)
    assert automl.X_holdout is None
    assert automl.y_holdout is None
    assert automl.passed_holdout_set is False

    match_text = "Holdout set size must be greater than 0 and less than 1. Set holdout set size to 0 to disable holdout set evaluation."
    with pytest.raises(
        ValueError,
        match=match_text,
    ):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            holdout_set_size=-0.1,
        )
