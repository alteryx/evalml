import numpy as np
import pandas as pd
import pytest

from evalml.automl import AutoMLSearch
from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.automl.callbacks import raise_error_callback
from evalml.automl.engine import CFEngine, DaskEngine, SequentialEngine
from evalml.problem_types import ProblemTypes, is_binary, is_multiclass, is_time_series
from evalml.tests.automl_tests.dask_test_utils import (
    DaskPipelineFast,
    DaskPipelineSlow,
    DaskPipelineWithFitError,
    DaskPipelineWithScoreError,
)
from evalml.tuners import SKOptTuner

# The engines to parametrize the AutoML tests over.  The process-level parallel tests
# are flaky.
engine_strs = ["cf_threaded", "dask_threaded"]


@pytest.fixture(scope="module")
def sequential_results(X_y_binary_cls):
    X, y = X_y_binary_cls

    seq_automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine="sequential",
    )
    seq_automl.search()
    sequential_rankings = seq_automl.full_rankings
    seq_results = sequential_rankings.drop(columns=["id"])
    return seq_results


@pytest.mark.parametrize(
    "engine_str",
    engine_strs,
)
def test_automl(
    engine_str,
    X_y_binary_cls,
    sequential_results,
):
    """Comparing the results of parallel and sequential AutoML to each other."""

    X, y = X_y_binary_cls
    par_automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine=engine_str,
    )
    par_automl.search()
    par_automl.close_engine()
    parallel_rankings = par_automl.full_rankings

    par_results = parallel_rankings.drop(columns=["id"])
    seq_results = sequential_results

    assert all(seq_results["pipeline_name"] == par_results["pipeline_name"])
    assert np.allclose(
        np.array(seq_results["mean_cv_score"]),
        np.array(par_results["mean_cv_score"]),
    )
    assert np.allclose(
        np.array(seq_results["validation_score"]),
        np.array(par_results["validation_score"]),
    )
    assert np.allclose(
        np.array(seq_results["percent_better_than_baseline"]),
        np.array(par_results["percent_better_than_baseline"]),
    )


@pytest.mark.parametrize(
    "engine_str",
    engine_strs,
)
def test_automl_max_iterations(
    engine_str,
    X_y_binary_cls,
):
    """Making sure that the max_iterations parameter limits the number of pipelines run."""

    X, y = X_y_binary_cls
    max_iterations = 4
    par_automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine=engine_str,
        max_iterations=max_iterations,
    )
    par_automl.search()
    par_automl.close_engine()
    parallel_rankings = par_automl.full_rankings

    seq_automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine="sequential",
        max_iterations=max_iterations,
    )
    seq_automl.search()
    sequential_rankings = seq_automl.full_rankings

    assert len(sequential_rankings) == len(parallel_rankings) == max_iterations


@pytest.mark.parametrize(
    "engine_str",
    engine_strs,
)
def test_automl_train_dask_error_callback(
    engine_str,
    X_y_binary_cls,
    caplog,
):
    """Make sure the pipeline training error message makes its way back from the workers."""
    caplog.clear()
    X, y = X_y_binary_cls

    pipelines = [DaskPipelineWithFitError({})]
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine=engine_str,
        max_iterations=2,
    )
    automl.allowed_pipelines = pipelines

    automl.train_pipelines(pipelines)
    assert "Train error for PipelineWithError: Yikes" in caplog.text
    automl.close_engine()


@pytest.mark.parametrize(
    "engine_str",
    engine_strs,
)
def test_automl_score_dask_error_callback(
    engine_str,
    X_y_binary_cls,
    caplog,
):
    """Make sure the pipeline scoring error message makes its way back from the workers."""
    caplog.clear()
    X, y = X_y_binary_cls

    pipelines = [DaskPipelineWithScoreError({})]
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine=engine_str,
        max_iterations=2,
    )
    automl.allowed_pipelines = pipelines

    automl.score_pipelines(pipelines, X, y, objectives=["Log Loss Binary", "F1", "AUC"])
    assert "Score error for PipelineWithError" in caplog.text
    automl.close_engine()


@pytest.mark.parametrize(
    "engine_str",
    engine_strs,
)
def test_automl_immediate_quit(
    engine_str,
    X_y_binary_cls,
    caplog,
):
    """Make sure the AutoMLSearch quits when error_callback is defined and does no further work."""
    caplog.clear()
    X, y = X_y_binary_cls

    pipelines = [
        DaskPipelineFast({}),
        DaskPipelineWithFitError({}),
        DaskPipelineSlow({}),
    ]
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine=engine_str,
        max_iterations=4,
        error_callback=raise_error_callback,
        optimize_thresholds=False,
    )
    automl.automl_algorithm = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        tuner_class=SKOptTuner,
        random_seed=0,
        n_jobs=-1,
        number_features=X.shape[1],
        pipelines_per_batch=5,
        ensembling=False,
        text_in_ensembling=False,
        search_parameters={},
    )
    automl.automl_algorithm._set_allowed_pipelines(pipelines)

    # Ensure the broken pipeline raises the error
    with pytest.raises(Exception, match="Yikes"):
        automl.search()

    # Make sure the automl algorithm stopped after the broken pipeline raised
    assert len(automl.full_rankings) < len(pipelines)
    assert DaskPipelineSlow.custom_name not in set(
        automl.full_rankings["pipeline_name"],
    )
    assert DaskPipelineWithFitError.custom_name not in set(
        automl.full_rankings["pipeline_name"],
    )
    automl.close_engine()


@pytest.mark.parametrize(
    "engine_str",
    engine_strs + ["sequential"],
)
def test_engine_can_use_str_name_for_convenience(engine_str, X_y_binary_cls):
    """Test to assert the proper engine is set for each provided convenience string."""
    X, y = X_y_binary_cls
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine=engine_str,
        optimize_thresholds=False,
    )
    if "sequential" in engine_str:
        assert isinstance(automl._engine, SequentialEngine)
    elif "cf" in engine_str:
        assert isinstance(automl._engine, CFEngine)
    elif "dask" in engine_str:
        assert isinstance(automl._engine, DaskEngine)
    automl.close_engine()


def test_automl_convenience_exception(X_y_binary_cls):
    X, y = X_y_binary_cls
    with pytest.raises(ValueError, match="is not a valid engine, please choose from"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine="bad_choice",
            optimize_thresholds=False,
        )


@pytest.mark.parametrize(
    "engine_str",
    engine_strs + ["cf_process"],
)
def test_automl_closes_engines(engine_str, X_y_binary_cls):
    X, y = X_y_binary_cls
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        engine=engine_str,
        optimize_thresholds=False,
    )
    automl.close_engine()
    assert automl._engine.is_closed


@pytest.mark.parametrize(
    "engine_str",
    engine_strs + ["sequential"],
)
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_score_pipelines_passes_X_train_y_train(
    problem_type,
    engine_str,
    X_y_based_on_pipeline_or_problem_type,
    ts_data,
    AutoMLTestEnv,
):
    if is_time_series(problem_type):
        X, _, y = ts_data(problem_type=problem_type)
    else:
        X, y = X_y_based_on_pipeline_or_problem_type(problem_type)

    half = X.shape[0] // 2
    X_train, y_train = pd.DataFrame(X[:half]), pd.Series(y[:half])
    X_test, y_test = pd.DataFrame(X[half:]), pd.Series(y[half:])

    if is_multiclass(problem_type) or is_binary(problem_type):
        y_train = y_train.astype("int64")
        y_test = y_test.astype("int64")

    automl = AutoMLSearch(
        X_train=X_train,
        y_train=y_train,
        problem_type=problem_type,
        max_iterations=5,
        optimize_thresholds=False,
        problem_configuration={
            "time_index": "date",
            "gap": 0,
            "forecast_horizon": 1,
            "max_delay": 1,
        },
        engine=engine_str,
    )

    env = AutoMLTestEnv(problem_type)
    with env.test_context(score_return_value={automl.objective.name: 3.12}):
        automl.search()

    with env.test_context(score_return_value={automl.objective.name: 3.12}):
        automl.score_pipelines(
            automl.allowed_pipelines,
            X_test,
            y_test,
            [automl.objective],
        )

    expected_X_train, expected_y_train = None, None
    if is_time_series(problem_type):
        expected_X_train, expected_y_train = X_train, y_train
    assert len(env.mock_score.mock_calls) == len(automl.allowed_pipelines)
    for mock_call in env.mock_score.mock_calls:
        if expected_X_train is not None:
            pd.testing.assert_frame_equal(mock_call[2]["X_train"], expected_X_train)
            pd.testing.assert_series_equal(mock_call[2]["y_train"], expected_y_train)
        else:
            assert mock_call[2]["X_train"] == expected_X_train
            assert mock_call[2]["y_train"] == expected_y_train
