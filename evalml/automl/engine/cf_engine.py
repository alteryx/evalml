from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    score_pipeline,
    train_pipeline,
)


class CFClient:
    """Custom CFClient API to match Dask's CFClient and allow context management."""

    def __init__(self, pool):
        """
        Arguments:
            pool(cf.ThreadPoolExecutor or cf.ProcessPoolExecutor): the resource pool
                to execute the futures work on.
        """
        self.pool = pool

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        pass

    def submit(self, *args, **kwargs):
        """Pass through to imitate Dask's Client API."""
        return self.pool.submit(*args, **kwargs)


class CFComputation(EngineComputation):
    """A Future-like wrapper around jobs created by the CFEngine."""

    def __init__(self, future):
        """
        Arguments:
            future(cf.Future): The concurrent.futures.Future that is desired
                to be executed.
        """
        self.work = future
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
        """
        Returns:
            bool: Returns whether computation was cancelled.
        """
        return self.work.cancelled()


class CFEngine(EngineBase):
    """The concurrent.futures (CF) engine"""

    def __init__(self, client):
        if not isinstance(client, CFClient):
            raise TypeError(
                f"Expected evalml.automl.engine.cf_engine.CFClient, received {type(client)}"
            )
        self.client = client
        self._data_futures_cache = {}

    def submit_evaluation_job(self, automl_config, pipeline, X, y) -> EngineComputation:
        """Send evaluation job to cluster.

        Arguments:
            automl_config: structure containing data passed from AutoMLSearch instance
            pipeline (pipeline.PipelineBase): pipeline to evaluate
            X (pd.DataFrame): input data for modeling
            y (pd.Series): target data for modeling
        Return:
            CFComputation: an object wrapping a reference to a future-like computation
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

    def submit_training_job(self, automl_config, pipeline, X, y) -> EngineComputation:
        """Send training job to cluster.

        Arguments:
            automl_config: structure containing data passed from AutoMLSearch instance
            pipeline (pipeline.PipelineBase): pipeline to train
            X (pd.DataFrame): input data for modeling
            y (pd.Series): target data for modeling
        Return:
            CFComputation: an object wrapping a reference to a future-like computation
                occurring in the resource pool
        """
        future = self.client.submit(
            train_pipeline, pipeline=pipeline, X=X, y=y, automl_config=automl_config
        )
        return CFComputation(future)

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
            CFComputation: a object wrapping a reference to a future-like computation
                occurring in the resource pool
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
        )
        computation = CFComputation(future)
        computation.meta_data["pipeline_name"] = pipeline.name
        return computation
