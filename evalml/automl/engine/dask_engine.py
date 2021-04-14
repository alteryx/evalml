import joblib
from dask.distributed import Client

from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    score_pipeline,
    train_pipeline
)


class DaskComputation(EngineComputation):
    """A Future-like wrapper around jobs created by the DaskEngine."""

    def __init__(self, dask_future):
        self.work = dask_future
        self.meta_data = {}

    def done(self):
        """Whether the computation is done."""
        return self.work.done()

    def get_result(self):
        """Gets the computation result.
        Will block until the computation is finished.

        Raises Exception: If computation fails. Returns traceback.
        """
        return self.work.result()

    def cancel(self):
        """Cancel the current computation."""
        return self.work.cancel()

    @property
    def is_cancelled(self):
        """Returns whether computation was cancelled."""
        return self.work.status


class DaskEngine(EngineBase):
    """The dask engine"""

    def __init__(self, client):
        if not isinstance(client, Client):
            raise TypeError(f"Expected dask.distributed.Client, received {type(client)}")
        self.client = client
        self._data_futures_cache = {}

    def send_data_to_cluster(self, X, y):
        """Send data to the cluster.

        The implementation uses caching so the data is only sent once. This follows
        dask best practices.

        Args:
            X (pd.DataFrame, ww.DataTable): input data for modeling
            y (pd.DataSeries, ww.DataColumn): target data for modeling
        Return:
            dask.Future: the modeling data
        """
        data_hash = joblib.hash(X), joblib.hash(y)
        if data_hash in self._data_futures_cache:
            X_future, y_future = self._data_futures_cache[data_hash]
            if not (X_future.cancelled() or y_future.cancelled()):
                return X_future, y_future
        self._data_futures_cache[data_hash] = self.client.scatter([X, y], broadcast=True)
        return self._data_futures_cache[data_hash]

    def submit_evaluation_job(self, automl_config, pipeline, X, y) -> EngineComputation:
        """Send evaluation job to cluster.

        Args:
            automl_config: structure containing data passed from AutoMLSearch instance
            pipeline (pipeline.PipelineBase): pipeline to evaluate
            X (pd.DataFrame, ww.DataTable): input data for modeling
            y (pd.DataSeries, ww.DataColumn): target data for modeling
        Return:
            DaskComputation: a object wrapping a reference to a future-like computation
                occurring in the dask cluster
        """
        logger = self.setup_job_log()
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(evaluate_pipeline, pipeline=pipeline,
                                         automl_config=automl_config,
                                         X=X,
                                         y=y,
                                         logger=logger)
        return DaskComputation(dask_future)

    def submit_training_job(self, automl_config, pipeline, X, y) -> EngineComputation:
        """Send training job to cluster.

        Args:
            automl_config: structure containing data passed from AutoMLSearch instance
            pipeline (pipeline.PipelineBase): pipeline to train
            X (pd.DataFrame, ww.DataTable): input data for modeling
            y (pd.DataSeries, ww.DataColumn): target data for modeling
        Return:
            DaskComputation: a object wrapping a reference to a future-like computation
                occurring in the dask cluster
        """
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(train_pipeline,
                                         pipeline=pipeline, X=X,
                                         y=y,
                                         optimize_thresholds=automl_config.optimize_thresholds,
                                         objective=automl_config.objective)
        return DaskComputation(dask_future)

    def submit_scoring_job(self, automl_config, pipeline, X, y, objectives) -> EngineComputation:
        """Send scoring job to cluster.

        Args:
            automl_config: structure containing data passed from AutoMLSearch instance
            pipeline (pipeline.PipelineBase): pipeline to train
            X (pd.DataFrame, ww.DataTable): input data for modeling
            y (pd.DataSeries, ww.DataColumn): target data for modeling
        Return:
            DaskComputation: a object wrapping a reference to a future-like computation
                occurring in the dask cluster
        """
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(score_pipeline, pipeline=pipeline,
                                         X=X, y=y, objectives=objectives)
        computation = DaskComputation(dask_future)
        computation.meta_data["pipeline_name"] = pipeline.name
        return computation
