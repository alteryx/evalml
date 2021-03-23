import joblib

from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    train_pipeline,
    score_pipeline
)


class DaskComputation(EngineComputation):
    """A Future-like wrapper around jobs created by the DaskEngine."""

    def __init__(self, dask_future):
        self.work = dask_future
        self.meta_data = {}

    def done(self) -> bool:
        """Is the computation done?"""
        return self.work.done()

    def get_result(self):
        """Get the computation result.
        Will block until the computation is finished.

        Raises Exception: If computation fails. Returns traceback.
        """
        return self.work.result()

    def cancel(self) -> None:
        """Cancel the current computation."""
        return self.work.cancel()


class DaskEngine(EngineBase):
    """The dask engine"""

    def __init__(self, client):
        self.client = client
        self.cache = {}

    def send_data_to_cluster(self, X, y):
        """Send data to the cluster.

        The implementation uses caching so the data is only sent once. This follows
        dask best practices.
        """
        data_hash = joblib.hash(X), joblib.hash(y)
        if data_hash in self.cache:
            return self.cache[data_hash]
        self.cache[data_hash] = self.client.scatter([X, y], broadcast=True)
        return self.cache[data_hash]

    def submit_evaluation_job(self, automl_data, pipeline, X, y) -> EngineComputation:
        logger = self.setup_job_log()
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(evaluate_pipeline, pipeline=pipeline,
                                         automl_data=automl_data,
                                         X=X,
                                         y=y,
                                         logger=logger)
        return DaskComputation(dask_future)

    def submit_training_job(self, automl_data, pipeline, X, y) -> EngineComputation:
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(train_pipeline,
                                         pipeline=pipeline, X=X,
                                         y=y,
                                         optimize_thresholds=automl_data.optimize_thresholds,
                                         objective=automl_data.objective)
        return DaskComputation(dask_future)

    def submit_scoring_job(self, automl_data, pipeline, X, y, objectives):
        X, y = self.send_data_to_cluster(X, y)
        dask_future = self.client.submit(score_pipeline, pipeline=pipeline,
                                         X=X, y=y, objectives=objectives)
        computation = DaskComputation(dask_future)
        computation.meta_data["pipeline_name"] = pipeline.name
        return computation
