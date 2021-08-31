import joblib
from dask.distributed import Client, LocalCluster

from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    score_pipeline,
    train_pipeline,
)


class DaskComputation(EngineComputation):
    """A Future-like wrapper around jobs created by the DaskEngine.

    Arguments:
        dask_future (callable): Computation to do.
    """

    def __init__(self, dask_future):
        self.work = dask_future
        self.meta_data = {}

    def done(self):
        """
        Returns:
            bool: Whether the computation is done.
        """
        return self.work.done()

    def get_result(self):
        """Gets the computation result.
        Will block until the computation is finished.

        Raises:
             Exception: If computation fails. Returns traceback.
        """
        return self.work.result()

    def cancel(self):
        """Cancel the current computation."""
        return self.work.cancel()

    @property
    def is_cancelled(self):
        """
        Returns:
            bool: Returns whether computation was cancelled.
        """
        return self.work.status


class DaskEngine(EngineBase):
    """The dask engine

    Arguments:
        cluster (None or dd.Client): If None, creates a local, threaded Dask client for processing.
            Defaults to None.
    """

    def __init__(self, cluster=None):
        if cluster is not None and not isinstance(cluster, (LocalCluster)):
            raise TypeError(
                f"Expected dask.distributed.Client, received {type(cluster)}"
            )
        elif cluster is None:
            cluster = LocalCluster(processes=False)
        self.cluster = cluster
        self.client = Client(self.cluster)
        self._data_futures_cache = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def send_data_to_cluster(self, X, y):
        """Send data to the cluster.

        The implementation uses caching so the data is only sent once. This follows
        dask best practices.

        Arguments:
            X (pd.DataFrame): input data for modeling
            y (pd.Series): target data for modeling
        Return:
            dask.Future: the modeling data
        """
        data_hash = joblib.hash(X), joblib.hash(y)
        if data_hash in self._data_futures_cache:
            X_future, y_future = self._data_futures_cache[data_hash]
            if not (X_future.cancelled() or y_future.cancelled()):
                return X_future, y_future
        self._data_futures_cache[data_hash] = self.client.scatter(
            [X, y], broadcast=True
        )
        return self._data_futures_cache[data_hash]

    def submit_evaluation_job(self, automl_config, pipeline, X, y) -> EngineComputation:
        """Send evaluation job to cluster.

        Arguments:
            automl_config: structure containing data passed from AutoMLSearch instance
            pipeline (pipeline.PipelineBase): pipeline to evaluate
            X (pd.DataFrame): input data for modeling
            y (pd.Series): target data for modeling
        Return:
            DaskComputation: a object wrapping a reference to a future-like computation
                occurring in the dask cluster
        """
        logger = self.setup_job_log()
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(
            evaluate_pipeline,
            pipeline=pipeline,
            automl_config=automl_config,
            X=X,
            y=y,
            logger=logger,
        )
        return DaskComputation(dask_future)

    def submit_training_job(self, automl_config, pipeline, X, y) -> EngineComputation:
        """Send training job to cluster.

        Arguments:
            automl_config: structure containing data passed from AutoMLSearch instance
            pipeline (pipeline.PipelineBase): pipeline to train
            X (pd.DataFrame): input data for modeling
            y (pd.Series): target data for modeling
        Return:
            DaskComputation: a object wrapping a reference to a future-like computation
                occurring in the dask cluster
        """
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(
            train_pipeline, pipeline=pipeline, X=X, y=y, automl_config=automl_config
        )
        return DaskComputation(dask_future)

    def submit_scoring_job(
        self, automl_config, pipeline, X, y, objectives
    ) -> EngineComputation:
        """Send scoring job to cluster.

        Arguments:
            automl_config: structure containing data passed from AutoMLSearch instance
            pipeline (pipeline.PipelineBase): pipeline to train
            X (pd.DataFrame): input data for modeling
            y (pd.Series): target data for modeling
        Return:
            DaskComputation: a object wrapping a reference to a future-like computation
                occurring in the dask cluster
        """
        # Get the schema before we lose it
        X_schema = X.ww.schema
        y_schema = y.ww.schema
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(
            score_pipeline,
            pipeline=pipeline,
            X=X,
            y=y,
            objectives=objectives,
            X_schema=X_schema,
            y_schema=y_schema,
        )
        computation = DaskComputation(dask_future)
        computation.meta_data["pipeline_name"] = pipeline.name
        return computation

    def close(self):
        """Closes the underlying cluster."""
        # TODO: Might want to rethink this if using something other than a LocalCluster.
        self.cluster.close()
        self.client.close()

    @property
    def is_closed(self):
        """Property that determines whether the Engine's Client's resources are shutdown."""
        return self.cluster.status.value == "closed"
