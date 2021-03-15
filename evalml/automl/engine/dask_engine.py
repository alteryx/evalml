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

    def submit_evaluation_job(self, automl_data, pipeline) -> EngineComputation:
        logger = self.setup_job_log()
        dask_future = self.client.submit(evaluate_pipeline, pipeline=pipeline,
                                         automl_data=automl_data,
                                         X=automl_data.X_train,
                                         y=automl_data.y_train,
                                         logger=logger)
        return DaskComputation(dask_future)

    def submit_training_job(self, automl_data, pipeline) -> EngineComputation:

        dask_future = self.client.submit(train_pipeline,
                                         pipeline=pipeline, X=automl_data.X_train,
                                         y=automl_data.y_train,
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
