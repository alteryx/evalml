from evalml.automl.engines import EngineBase
from dask.distributed import Client, as_completed, wait


class DaskEngine(EngineBase):
    def __init__(self, cluster):
        self.name = "Dask Engine"
        if not cluster:
            cluster = Client()
        elif not isinstance(cluster, Client):
            raise ValueError("Dask cluster was not passed into the engine")
        self.dask_cluster = cluster
        super().__init__()

    def load_data(self, X, y, search):
        self.X_future = self.dask_cluster.scatter(X)
        self.y_future = self.dask_cluster.scatter(y)
        super().load_data(X, y, search)

    def evaluate_batch(self, pipeline_batch):
        fitted_pipelines = []
        evaluation_results = []
        for pipeline in pipeline_batch:
            self.log_pipeline(pipeline)
        pipeline_futures = self.dask_cluster.map(self.search._evaluate, pipeline_batch, X=self.X_future, y=self.y_future)
        for pipeline_future in as_completed(pipeline_futures):
            fitted_pipeline, evaluation_result = pipeline_future.result()
            fitted_pipelines.append(fitted_pipeline)
            evaluation_results.append(evaluation_result)
        wait(pipeline_futures)
        return fitted_pipelines, evaluation_results

    def evaluate_pipeline(self, pipeline):
        self.log_pipeline(pipeline)
        pipeline_future = self.dask_cluster.submit(self.search._evaluate, pipeline=pipeline, X=self.X_future, y=self.y_future)
        fitted_pipeline, evaluation_result = pipeline_future.result()
        return [fitted_pipeline], [evaluation_result]
