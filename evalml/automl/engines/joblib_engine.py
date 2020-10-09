from evalml.automl.engines import EngineBase
from joblib import Parallel


class JoblibEngine(EngineBase):
    def __init__(self, workers):
        self.name = "Joblib Engine"
        if not isinstance(workers, int):
            raise ValueError("workers must be an int")
        self.executer = Parallel(n_jobs=workers)
        super().__init__()

    def evaluate_batch(self, pipeline_batch):
        fitted_pipelines = []
        evaluation_results = []
        for pipeline in pipeline_batch:
            self.log_pipeline(pipeline)
        results = self.executer(self.search._evaluate(pipeline, self.X, self.y) for pipeline in pipeline_batch)
        for result in results:
            fitted_pipeline, evaluation_result = result
            fitted_pipelines.append(fitted_pipeline)
            evaluation_results.append(evaluation_result)
        return fitted_pipelines, evaluation_results

    def evaluate_pipeline(self, pipeline):
        self.log_pipeline(pipeline)
        fitted_pipeline, evaluation_result = self.search._evaluate(pipeline, self.X, self.y)
        return [fitted_pipeline], [evaluation_result]
