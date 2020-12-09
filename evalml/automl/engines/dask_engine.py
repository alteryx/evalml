from evalml.automl.engines import EngineBase
from dask.distributed import Client, as_completed


class DaskEngine(EngineBase):
    def __init__(self, dask_client=None):
        super().__init__()
        self.name = "Dask Engine"
        self.client = dask_client if dask_client else Client()

    def load_data(self, X, y):
        self.X_future = self.client.scatter(X)
        self.y_future = self.client.scatter(y)
        super().load_data(X, y)

    def evaluate_batch(self, pipeline_batch, result_callback=None, log_pipeline=False):
        """ Distributes a batch of pipelines to be evaluated by Dask workers. Pipeline evaluation order is not guaranteed and there is currently no graceful termination.

        Arguments:
            pipeline_batch (list(PipelineBase)): A batch of pipelines to be fitted and evaluated.
            result_callback (callable): Function called once the pipeline is finished evaluating to store the results. If None, results will only be returned.
            log_pipeline (bool): If True, log the pipeline and relevant information before evaluation.

        Returns:
            list(PipelineBase), list(dict), list(PipelineBase): A list of evaluated pipelines, the results of the evaluated pipelines, and the remaining pipelines in the batch that were not evaluated.
        """
        super().evaluate_pipeline()
        completed_pipelines = []
        evaluation_results = []
        for pipeline in pipeline_batch:
            if self.automl.start_iteration_callback:
                self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self.automl)
        pipeline_futures = self.client.map(self._compute_cv_scores, pipeline_batch, automl=self.automl, X=self.X_future, y=self.y_future)
        for future in as_completed(pipeline_futures):
            pipeline, evaluation_result = future.result()
            if log_pipeline:
                self.log_pipeline(pipeline)
            if result_callback:
                self._add_result_callback(result_callback, pipeline, evaluation_result)
            completed_pipelines.append(pipeline)
            evaluation_results.append(evaluation_result)
        return completed_pipelines, evaluation_results, []

    def evaluate_pipeline(self, pipeline, result_callback=None, log_pipeline=False):
        """ Evaluates a single pipeline by sending it to a Dask worker.

        Arguments:
            pipeline_batch (PipelineBase): A pipeline to be fitted and evaluated.
            result_callback (callable): Function called once the pipeline is finished evaluating to store the results. If None, results will only be returned.
            log_pipeline (bool): If True, log the pipeline and relevant information before evaluation.

        Returns:
            PipelineBase, dict: The evaluated pipeline and dictionary of the results results. If the search was terminated early, then results will be empty.
        """
        super().evaluate_pipeline()
        try:
            if log_pipeline:
                self.log_pipeline(pipeline)
            if self.automl.start_iteration_callback:
                self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)
            pipeline, result = self.client.submit(self._compute_cv_scores, pipeline, automl=self.automl, X=self.X_future, y=self.y_future)
            if result_callback:
                self._add_result_callback(result_callback, pipeline, result)
            return pipeline, result
        except KeyboardInterrupt:
            pipeline_batch = self.automl._handle_keyboard_interrupt([], pipeline)
            return pipeline_batch, []
