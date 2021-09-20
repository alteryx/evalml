"""Custom CFClient API to match Dask's CFClient and allow context management."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    score_pipeline,
    train_pipeline,
)


class CFClient:
    """Custom CFClient API to match Dask's CFClient and allow context management.

    Args:
        pool(cf.ThreadPoolExecutor or cf.ProcessPoolExecutor): The resource pool to execute the futures work on.
    """

    def __init__(self, pool):
        self.pool = pool

    def __enter__(self):
        """Enter runtime context."""
        return self

    def __exit__(self, typ, value, traceback):
        """Exit runtime context."""
        pass

    def submit(self, *args, **kwargs):
        """Pass through to imitate Dask's Client API."""
        return self.pool.submit(*args, **kwargs)

    def close(self):
        """Closes the underlying Executor."""
        self.pool.shutdown()

    @property
    def is_closed(self):
        """Property that determines whether the Engine's Client's resources are closed."""
        if isinstance(self.pool, ProcessPoolExecutor):
            return self.pool._shutdown_thread
        elif isinstance(self.pool, ThreadPoolExecutor):
            return self.pool._shutdown


class CFComputation(EngineComputation):
    """A Future-like wrapper around jobs created by the CFEngine.

    Args:
        future(cf.Future): The concurrent.futures.Future that is desired to be executed.
    """

    def __init__(self, future):
        self.work = future
        self.meta_data = {}

    def done(self):
        """Returns whether the computation is done."""
        return self.work.done()

    def get_result(self):
        """Gets the computation result. Will block until the computation is finished.

        Raises:
            Exception: If computation fails. Returns traceback.
            cf.TimeoutError: If computation takes longer than default timeout time.
            cf.CancelledError: If computation was canceled before completing.

        Returns:
            The result of the requested job.
        """
        return self.work.result()

    def cancel(self):
        """Cancel the current computation.

        Returns:
            bool: False if the call is currently being executed or finished running
              and cannot be cancelled.  True if the call can be canceled.
        """
        return self.work.cancel()

    @property
    def is_cancelled(self):
        """Returns whether computation was cancelled."""
        return self.work.cancelled()


class CFEngine(EngineBase):
    """The concurrent.futures (CF) engine.

    Args:
        client (None or CFClient): If None, creates a threaded pool for processing. Defaults to None.
    """

    def __init__(self, client=None):
        if client is not None and not isinstance(client, CFClient):
            raise TypeError(
                f"Expected evalml.automl.engine.cf_engine.CFClient, received {type(client)}"
            )
        elif client is None:
            client = CFClient(ThreadPoolExecutor())
        self.client = client
        self._data_futures_cache = {}

    def submit_evaluation_job(self, automl_config, pipeline, X, y):
        """Send evaluation job to cluster.

        Args:
            automl_config: Structure containing data passed from AutoMLSearch instance.
            pipeline (pipeline.PipelineBase): Pipeline to evaluate.
            X (pd.DataFrame): Input data for modeling.
            y (pd.Series): Target data for modeling.

        Returns:
            CFComputation: An object wrapping a reference to a future-like computation
                occurring in the resource pool
        """
        logger = self.setup_job_log()
        future = self.client.submit(
            evaluate_pipeline,
            pipeline=pipeline,
            automl_config=automl_config,
            X=X,
            y=y,
            logger=logger,
        )
        return CFComputation(future)

    def submit_training_job(self, automl_config, pipeline, X, y):
        """Send training job to cluster.

        Args:
            automl_config: Structure containing data passed from AutoMLSearch instance.
            pipeline (pipeline.PipelineBase): Pipeline to train.
            X (pd.DataFrame): Input data for modeling.
            y (pd.Series): Target data for modeling.

        Returns:
            CFComputation: An object wrapping a reference to a future-like computation
                occurring in the resource pool
        """
        future = self.client.submit(
            train_pipeline, pipeline=pipeline, X=X, y=y, automl_config=automl_config
        )
        return CFComputation(future)

    def submit_scoring_job(
        self, automl_config, pipeline, X, y, objectives, X_train=None, y_train=None
    ):
        """Send scoring job to cluster.

        Args:
            automl_config: Structure containing data passed from AutoMLSearch instance.
            pipeline (pipeline.PipelineBase): Pipeline to train.
            X (pd.DataFrame): Input data for modeling.
            y (pd.Series): Target data for modeling.
            X_train (pd.DataFrame): Training features. Used for feature engineering in time series.
            y_train (pd.Series): Training target. Used for feature engineering in time series.
            objectives (list[ObjectiveBase]): Objectives to score on.

        Returns:
            CFComputation: An object wrapping a reference to a future-like computation
                occurring in the resource pool.
        """
        # Get the schema before we lose it
        X_schema = X.ww.schema
        y_schema = y.ww.schema
        future = self.client.submit(
            score_pipeline,
            pipeline=pipeline,
            X=X,
            y=y,
            objectives=objectives,
            X_schema=X_schema,
            y_schema=y_schema,
            X_train=X_train,
            y_train=y_train,
        )
        computation = CFComputation(future)
        computation.meta_data["pipeline_name"] = pipeline.name
        return computation

    def close(self):
        """Function to properly shutdown the Engine's Client's resources."""
        self.client.close()

    @property
    def is_closed(self):
        """Property that determines whether the Engine's Client's resources are shutdown."""
        return self.client.is_closed
