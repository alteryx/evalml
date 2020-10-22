from evalml.automl.engines import EngineBase


class SequentialEngine(EngineBase):
    """The default engine for the AutoML search. Evaluates pipelines locally and pipeline batches sequentially."""
    def __init__(self):
        super().__init__()
        self.name = "Sequential Engine"

    def evaluate_batch(self, pipeline_batch, result_callback=None, log_pipeline=False):
        """ Evaluates a batch of pipelines in order.

        Arguments:
            pipeline_batch (list(PipelineBase)): A batch of pipelines to be fitted and evaluated.
            result_callback (callable): Function called once the pipeline is finished evaluating to store the results. If None, results will only be returned.
            log_pipeline (bool): If True, log the pipeline and relevant information before evaluation.

        Returns:
            list(PipelineBase), list(dict), list(PipelineBase): A list of evaluated pipelines, the results of the evaluated pipelines, and the remaining pipelines in the batch that were not evaluated.
        """
        super().evaluate_batch()
        completed_pipelines = []
        evaluation_results = []
        while len(pipeline_batch) > 0:
            pipeline = pipeline_batch.pop()
            try:
                if not self.automl._check_stopping_condition(self.automl._start):
                    return completed_pipelines, evaluation_results, []
                if log_pipeline:
                    self.log_pipeline(pipeline)
                if self.automl.start_iteration_callback:
                    self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)
                pipeline, evaluation_result = self._compute_cv_scores(pipeline, self.automl, self.X, self.y)
                if result_callback:
                    self._add_result_callback(result_callback, pipeline, evaluation_result)
                completed_pipelines.append(pipeline)
                evaluation_results.append(evaluation_result)
            except KeyboardInterrupt:
                pipeline_batch = self._handle_keyboard_interrupt(pipeline_batch, pipeline)
                if pipeline_batch == []:
                    return completed_pipelines, evaluation_results, pipeline_batch
        return completed_pipelines, evaluation_results, pipeline_batch

    def evaluate_pipeline(self, pipeline, result_callback=None, log_pipeline=False):
        """ Evaluates a single pipeline.

        Arguments:
            pipeline_batch (PipelineBase): A pipelines to be fitted and evaluated.
            result_callback (callable): Function called once the pipeline is finished evaluating to store the results. If None, results will only be returned.
            log_pipeline (bool): If True, log the pipeline and relevant information before evaluation.

        Returns:
            PipelineBase, dict: The evaluated pipeline and dictionary of the results results. If the search was terminated early, then results will be empty.
        """
        super().evaluate_pipeline()
        try:
            evaluation_result = None
            if log_pipeline:
                self.log_pipeline(pipeline)
            if self.automl.start_iteration_callback:
                self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)
            pipeline, evaluation_result = self._compute_cv_scores(pipeline, self.automl, self.X, self.y)
            if result_callback:
                self._add_result_callback(result_callback, pipeline, evaluation_result)
            return pipeline, evaluation_result
        except KeyboardInterrupt:
            pipeline = self._handle_keyboard_interrupt([], pipeline)
            return pipeline, []
