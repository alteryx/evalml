from dask.distributed import Client, as_completed

from evalml.automl.engines import EngineBase, EngineResult


class DaskEngine(EngineBase):
    def __init__(self, dask_client=None):
        super().__init__()
        self.name = "Dask Engine"
        self.client = dask_client if dask_client else Client()

    def load_data(self, X, y):
        self.X_future = self.client.scatter(X)
        self.y_future = self.client.scatter(y)
        super().load_data(X, y)

    def evaluate_batch(self, pipeline_batch, result_callback=None, log_pipeline=False, ignore_stopping_condition=False):
        """ Evaluates a batch of pipelines in order.

        Arguments:
            pipeline_batch (list(PipelineBase)): A batch of pipelines to be fitted and evaluated.
            result_callback (callable): Function called once the pipeline is finished evaluating to store the results. If None, results will only be returned.
            log_pipeline (bool): If True, log the pipeline and relevant information before evaluation.
            ignore_stopping_condition (bool): If True, will add pipelines regardless of stopping condition.
                If False, calls `_check_stopping_condition` to determine if additional pipelines should be evaluated. Default is False.

        Returns:
            EngineResult: An engine result object with completed pipelines, results, and unprocessed pipelines.
        """
        super().evaluate_batch()
        result = EngineResult()
        if not ignore_stopping_condition and not self.automl._check_stopping_condition(self.automl._start):
            result.set_early_stop(None, pipeline_batch)
            return result
        if self.automl.start_iteration_callback:
            for pipeline in pipeline_batch:
                self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self.automl)
        
        dask_pipelines = []
        while len(pipeline_batch) > 0:
            dask_pipelines.append(pipeline_batch.pop())
        
        pipeline_futures = self.client.map(self._compute_cv_scores, dask_pipelines, automl=self.automl, X=self.X_future, y=self.y_future)

        for future in as_completed(pipeline_futures):
            pipeline, evaluation_result = future.result()
            if log_pipeline:
                self.log_pipeline(pipeline)
            if result_callback:
                self._add_result_callback(result_callback, pipeline, evaluation_result)
            result.add_result(pipeline, evaluation_result)
        return result
