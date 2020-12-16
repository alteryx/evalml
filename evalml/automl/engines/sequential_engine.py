from evalml.automl.engines import EngineBase


class SequentialEngine(EngineBase):
    """The default engine for the AutoML search. Evaluates pipelines locally and pipeline batches sequentially."""
    def __init__(self):
        super().__init__()
        self.name = "Sequential Engine"

    def evaluate_batch(self, pipeline_batch, result_callback=None, log_pipeline=False, ignore_stopping_condition=False):
        """ Evaluates a batch of pipelines in order.

        Arguments:
            pipeline_batch (list(PipelineBase)): A batch of pipelines to be fitted and evaluated.
            result_callback (callable): Function called once the pipeline is finished evaluating to store the results. If None, results will only be returned.
            log_pipeline (bool): If True, log the pipeline and relevant information before evaluation.
            ignore_stopping_condition (bool): If True, will add pipelines regardless of stopping condition. 
                If False, calls `_check_stopping_condition` to determine if additional pipelines should be evaluated. Default is False.

        Returns:
            list(PipelineBase), list(dict), list(PipelineBase): A list of evaluated pipelines, the results of the evaluated pipelines, and the remaining pipelines in the batch that were not evaluated.
        """
        super().evaluate_batch()
        completed_pipelines = []
        evaluation_results = []
        while len(pipeline_batch) > 0:
            pipeline = pipeline_batch.pop()
            try:
                if not ignore_stopping_condition and not self.automl._check_stopping_condition(self.automl._start):
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
