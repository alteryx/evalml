import joblib

from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    train_pipeline
)


class DaskComputation(EngineComputation):

    def __init__(self, dask_future):
        self.work = dask_future
        self.meta_data = {}

    def done(self) -> bool:
        return self.work.done()

    def get_result(self):
        return self.work.result()

    def cancel(self) -> None:
        return self.work.cancel()


class DaskEngine(EngineBase):
    """The dask engine"""

    def __init__(self, client):
        self.client = client
        self.cache = {}

    def get_scattered_data(self, X, y):
        data_hash = joblib.hash(X), joblib.hash(y)
        if data_hash in self.cache:
            return self.cache[data_hash]
        self.cache[data_hash] = self.client.scatter([X, y], broadcast=True)
        return self.cache[data_hash]

    def submit_evaluation_job(self, automl_data, pipeline, X, y) -> EngineComputation:
        logger = self.setup_job_log()
        X, y = self.get_scattered_data(X, y)
        dask_future = self.client.submit(evaluate_pipeline, pipeline=pipeline,
                                         automl_data=automl_data,
                                         X=X,
                                         y=y,
                                         logger=logger)
        return DaskComputation(dask_future)

    def submit_training_job(self, automl_data, pipeline, X, y) -> EngineComputation:
        X, y = self.get_scattered_data(X, y)
        dask_future = self.client.submit(train_pipeline,
                                         pipeline=pipeline, X=X,
                                         y=y,
                                         optimize_thresholds=automl_data.optimize_thresholds,
                                         objective=automl_data.objective)
        return DaskComputation(dask_future)

    def submit_scoring_job(self, automl_data, pipeline, X, y, objectives):
        def score_pipeline():
            return pipeline.score(X, y, objectives)

        dask_future = self.client.submit(score_pipeline)
        computation = DaskComputation(dask_future)
        computation.meta_data["pipeline_name"] = pipeline.custom_name
        return computation
