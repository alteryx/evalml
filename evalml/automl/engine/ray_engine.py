from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    train_pipeline
)


class RayComputation(EngineComputation):

    def __init__(self, ray, ray_remote):
        self.ray = ray
        self.remote = ray_remote
        self.meta_data = {}

    def done(self):
        completed, not_completed = self.ray.wait([self.remote], timeout=0.01)
        return len(completed) > 0

    def get_result(self):
        return self.ray.get(self.remote)

    def cancel(self):
        self.ray.cancel(self.remote)


class RayEngine(EngineBase):
    """The ray engine"""

    def __init__(self, client):
        self.client = client
        self.cache = {}
        self.ray_evaluate_pipeline = self.client.remote(evaluate_pipeline).remote
        self.ray_train_pipeline = self.client.remote(train_pipeline).remote

        def score_pipeline(pipeline, X, y, objectives):
            return pipeline.score(X, y, objectives)

        self.ray_score_pipeline = self.client.remote(score_pipeline).remote

    def submit_evaluation_job(self, automl_data, pipeline, X, y) -> EngineComputation:
        logger = self.setup_job_log()
        ray_obj_ref = self.ray_evaluate_pipeline(pipeline=pipeline,
                                                 automl_data=automl_data,
                                                 X=X,
                                                 y=y,
                                                 logger=logger)
        return RayComputation(self.client, ray_obj_ref)

    def submit_training_job(self, automl_data, pipeline, X, y) -> EngineComputation:
        ray_obj_ref = self.ray_train_pipeline(pipeline=pipeline, X=X,
                                              y=y, optimize_thresholds=automl_data.optimize_thresholds,
                                              objective=automl_data.objective)
        return RayComputation(self.client, ray_obj_ref)

    def submit_scoring_job(self, automl_data, pipeline, X, y, objectives):
        ray_obj_ref = self.ray_score_pipeline(pipeline=pipeline, X=X, y=y, objectives=objectives)
        computation = RayComputation(self.client, ray_obj_ref)
        computation.meta_data["pipeline_name"] = pipeline.custom_name
        return computation
