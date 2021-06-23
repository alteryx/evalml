import numpy as np
import pytest
from dask.distributed import Client, LocalCluster

from evalml.automl import AutoMLSearch
from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.automl.callbacks import raise_error_callback
from evalml.automl.engine import DaskEngine, SequentialEngine
from evalml.tests.automl_tests.dask_test_utils import (
    TestPipelineFast,
    TestPipelineSlow,
    TestPipelineWithFitError,
    TestPipelineWithScoreError,
)
from evalml.tuners import SKOptTuner


@pytest.fixture
def sequential_engine():
    return SequentialEngine()


@pytest.fixture(scope="module")
def cluster():
    dask_cluster = LocalCluster(
        n_workers=1, threads_per_worker=2, dashboard_address=None
    )
    yield dask_cluster
    dask_cluster.close()


def test_automl(X_y_binary_cls, cluster, sequential_engine):
    """Comparing the results of parallel and sequential AutoML to each other."""
    with Client(cluster) as client:
        parallel_engine = DaskEngine(client)
        X, y = X_y_binary_cls
        par_automl = AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", engine=parallel_engine
        )
        par_automl.search()
        parallel_rankings = par_automl.full_rankings

        seq_automl = AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", engine=sequential_engine
        )
        seq_automl.search()
        sequential_rankings = seq_automl.full_rankings

        par_results = parallel_rankings.drop(columns=["id"])
        seq_results = sequential_rankings.drop(columns=["id"])

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


def test_automl_max_iterations(X_y_binary_cls, cluster, sequential_engine):
    """Making sure that the max_iterations parameter limits the number of pipelines run."""

    X, y = X_y_binary_cls
    with Client(cluster) as client:
        parallel_engine = DaskEngine(client)

        max_iterations = 4
        par_automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=parallel_engine,
            max_iterations=max_iterations,
        )
        par_automl.search()
        parallel_rankings = par_automl.full_rankings

        seq_automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=sequential_engine,
            max_iterations=max_iterations,
        )
        seq_automl.search()
        sequential_rankings = seq_automl.full_rankings

        assert len(sequential_rankings) == len(parallel_rankings) == max_iterations


def test_automl_train_dask_error_callback(X_y_binary_cls, cluster, caplog):
    """Make sure the pipeline training error message makes its way back from the workers."""
    caplog.clear()
    with Client(cluster) as client:
        parallel_engine = DaskEngine(client)
        X, y = X_y_binary_cls

        pipelines = [TestPipelineWithFitError({})]
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=parallel_engine,
            max_iterations=2,
        )
        automl.allowed_pipelines = pipelines

        automl.train_pipelines(pipelines)
        assert "Train error for PipelineWithError: Yikes" in caplog.text


def test_automl_score_dask_error_callback(X_y_binary_cls, cluster, caplog):
    """Make sure the pipeline scoring error message makes its way back from the workers."""
    caplog.clear()
    with Client(cluster) as client:
        parallel_engine = DaskEngine(client)

        X, y = X_y_binary_cls
        pipelines = [TestPipelineWithScoreError({})]
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=parallel_engine,
            max_iterations=2,
        )
        automl.allowed_pipelines = pipelines

        automl.score_pipelines(
            pipelines, X, y, objectives=["Log Loss Binary", "F1", "AUC"]
        )
        assert "Score error for PipelineWithError" in caplog.text


def test_automl_immediate_quit(X_y_binary_cls, cluster, caplog):
    """Make sure the AutoMLSearch quits when error_callback is defined and does no further work."""
    caplog.clear()
    X, y = X_y_binary_cls
    with Client(cluster) as client:
        parallel_engine = DaskEngine(client)

        pipelines = [
            TestPipelineFast({}),
            TestPipelineWithFitError({}),
            TestPipelineSlow({}),
        ]
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=parallel_engine,
            max_iterations=4,
            error_callback=raise_error_callback,
            optimize_thresholds=False,
        )
        automl._automl_algorithm = IterativeAlgorithm(
            max_iterations=4,
            allowed_pipelines=pipelines,
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

        # Ensure the broken pipeline raises the error
        with pytest.raises(Exception, match="Yikes"):
            automl.search()

        # Make sure the automl algorithm stopped after the broken pipeline raised
        assert len(automl.full_rankings) < len(pipelines)
        assert TestPipelineFast.custom_name in set(
            automl.full_rankings["pipeline_name"]
        )
        assert TestPipelineSlow.custom_name not in set(
            automl.full_rankings["pipeline_name"]
        )
        assert TestPipelineWithFitError.custom_name not in set(
            automl.full_rankings["pipeline_name"]
        )
