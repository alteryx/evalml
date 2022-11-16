import pandas as pd
import pytest

from evalml import AutoMLSearch
from evalml.automl.automl_algorithm import DefaultAlgorithm, IterativeAlgorithm
from evalml.exceptions import ObjectiveNotFoundError
from evalml.objectives import MeanSquaredLogError, RootMeanSquaredLogError
from evalml.pipelines import PipelineBase, TimeSeriesRegressionPipeline
from evalml.pipelines.components.utils import get_estimators
from evalml.preprocessing import TimeSeriesSplit


def test_init(X_y_regression):
    X, y = X_y_regression

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        objective="R2",
        max_iterations=3,
        n_jobs=1,
    )
    automl.search()

    assert isinstance(automl.automl_algorithm, DefaultAlgorithm)
    assert automl.n_jobs == 1
    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)
    automl.best_pipeline.predict(X)

    # test with dataframes
    automl = AutoMLSearch(
        pd.DataFrame(X),
        pd.Series(y),
        problem_type="regression",
        objective="R2",
        max_iterations=3,
        n_jobs=1,
    )
    automl.search()

    assert isinstance(automl.rankings, pd.DataFrame)
    assert isinstance(automl.full_rankings, pd.DataFrame)
    assert isinstance(automl.best_pipeline, PipelineBase)
    automl.best_pipeline.predict(X)
    assert isinstance(automl.get_pipeline(0), PipelineBase)

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        objective="R2",
        max_iterations=3,
        n_jobs=1,
        automl_algorithm="iterative",
    )

    assert isinstance(automl.automl_algorithm, IterativeAlgorithm)

    with pytest.raises(ValueError, match="Please specify a valid automl algorithm."):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
            objective="R2",
            max_iterations=3,
            n_jobs=1,
            automl_algorithm="not_valid",
        )


def test_callback(X_y_regression):
    X, y = X_y_regression

    counts = {
        "start_iteration_callback": 0,
        "add_result_callback": 0,
    }

    def start_iteration_callback(pipeline, automl_obj, counts=counts):
        counts["start_iteration_callback"] += 1

    def add_result_callback(results, trained_pipeline, automl_obj, counts=counts):
        counts["add_result_callback"] += 1

    max_iterations = 3
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        objective="R2",
        max_iterations=max_iterations,
        start_iteration_callback=start_iteration_callback,
        add_result_callback=add_result_callback,
        n_jobs=1,
        automl_algorithm="iterative",
    )
    automl.search()

    assert counts["start_iteration_callback"] == len(get_estimators("regression")) + 1
    assert counts["add_result_callback"] == max_iterations


def test_random_seed(X_y_regression):
    X, y = X_y_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        objective="R2",
        random_seed=0,
        n_jobs=1,
    )
    automl.search()

    automl_1 = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        objective="R2",
        random_seed=0,
        n_jobs=1,
    )
    automl_1.search()

    # need to use assert_frame_equal as R2 could be different at the 10+ decimal
    assert pd.testing.assert_frame_equal(automl.rankings, automl_1.rankings) is None


def test_categorical_regression(X_y_categorical_regression):
    X, y = X_y_categorical_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        objective="R2",
        random_seed=0,
        n_jobs=1,
    )
    automl.search()
    assert not automl.rankings["mean_cv_score"].isnull().any()


def test_plot_iterations_max_iterations(X_y_regression, go):

    X, y = X_y_regression

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        max_iterations=3,
        n_jobs=1,
    )
    automl.search()
    plot = automl.plot.search_iteration_plot()
    plot_data = plot.data[0]
    x = pd.Series(plot_data["x"])
    y = pd.Series(plot_data["y"])

    assert isinstance(plot, go.Figure)
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) == 3
    assert len(y) == 3


def test_plot_iterations_max_time(AutoMLTestEnv, X_y_regression, go):

    X, y = X_y_regression

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        max_time=2,
        random_seed=1,
        n_jobs=1,
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 0.2}):
        automl.search()
    plot = automl.plot.search_iteration_plot()
    plot_data = plot.data[0]
    x = pd.Series(plot_data["x"])
    y = pd.Series(plot_data["y"])

    assert isinstance(plot, go.Figure)
    assert x.is_monotonic_increasing
    assert y.is_monotonic_increasing
    assert len(x) > 0
    assert len(y) > 0


def test_log_metrics_only_passed_directly(X_y_regression):
    X, y = X_y_regression
    with pytest.raises(
        ObjectiveNotFoundError,
        match="RootMeanSquaredLogError is not a valid Objective!",
    ):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
            additional_objectives=["RootMeanSquaredLogError", "MeanSquaredLogError"],
        )

    ar = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        additional_objectives=[RootMeanSquaredLogError(), MeanSquaredLogError()],
    )
    assert ar.additional_objectives[0].name == "Root Mean Squared Log Error"
    assert ar.additional_objectives[1].name == "Mean Squared Log Error"


@pytest.mark.parametrize("freq", ["D", "MS"])
def test_automl_supports_time_series_regression(freq, AutoMLTestEnv, ts_data):
    X, _, y = ts_data(freq=freq)

    configuration = {
        "time_index": "date",
        "gap": 0,
        "max_delay": 0,
        "forecast_horizon": 6,
        "delay_target": False,
        "delay_features": True,
    }

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="time series regression",
        problem_configuration=configuration,
        max_batches=2,
    )
    env = AutoMLTestEnv("time series regression")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    assert isinstance(automl.data_splitter, TimeSeriesSplit)
    seen_with_decomp = []

    dt = configuration.pop("time_index")
    for result in automl.results["pipeline_results"].values():
        assert result["pipeline_class"] == TimeSeriesRegressionPipeline

        if result["id"] == 0:
            continue
        if "ARIMA Regressor" in result["parameters"]:
            dt_ = result["parameters"]["ARIMA Regressor"].pop("time_index")
            assert "DateTime Featurizer" not in result["parameters"].keys()
        elif "Prophet Regressor" in result["parameters"]:
            dt_ = result["parameters"]["Prophet Regressor"].pop("time_index")
            assert "DateTime Featurizer" not in result["parameters"].keys()
            assert "Time Series Featurizer" in result["parameters"].keys()
        else:
            dt_ = result["parameters"]["Time Series Featurizer"].pop("time_index")
        assert dt == dt_
        for param_key, param_val in configuration.items():
            assert (
                result["parameters"]["Time Series Featurizer"][param_key]
                == configuration[param_key]
            )
            assert (
                result["parameters"]["pipeline"][param_key] == configuration[param_key]
            )
        pipeline = result["pipeline_name"][:10]
        if pipeline not in seen_with_decomp:
            assert "STL Decomposer" in result["parameters"]
            seen_with_decomp.append(pipeline)
            dt_ = result["parameters"]["STL Decomposer"].pop("time_index")
            assert dt == dt_


@pytest.mark.parametrize(
    "sampler_method",
    [None, "auto", "Undersampler", "Oversampler"],
)
def test_automl_regression_no_sampler(sampler_method, X_y_regression):
    X, y = X_y_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        sampler_method=sampler_method,
    )
    for pipeline in automl.allowed_pipelines:
        assert not any("sampler" in c.name for c in pipeline.component_graph)
