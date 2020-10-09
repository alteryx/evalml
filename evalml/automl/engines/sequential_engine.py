from evalml.automl.engines import EngineBase


class SequentialEngine(EngineBase):
    def __init__(self):
        super().__init__()
        self.name = "Sequential Engine"

    def evaluate_batch(self, pipeline_batch):
        fitted_pipelines = []
        evaluation_results = []
        for pipeline in pipeline_batch:
            self.log_pipeline(pipeline)
            fitted_pipeline, evaluation_result = self.search._evaluate(pipeline, self.X, self.y)
            fitted_pipelines.append(fitted_pipeline)
            evaluation_results.append(evaluation_result)
        return fitted_pipelines, evaluation_results

    def evaluate_pipeline(self, pipeline):
        self.log_pipeline(pipeline)
        fitted_pipeline, evaluation_result = self.search._evaluate(pipeline, self.X, self.y)
        return [fitted_pipeline], [evaluation_result]
