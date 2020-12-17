from evalml.automl.engines import EngineBase
from evalml.automl.engines import EngineResult


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
            EngineResult: An engine result object with completed pipelines, results, and unprocessed pipelines.
        """
        super().evaluate_batch()
        result = EngineResult()
        pipeline = None
        while len(pipeline_batch) > 0:
            pipeline = pipeline_batch.pop()
            try:
                if not ignore_stopping_condition and not self.automl._check_stopping_condition(self.automl._start):
                    result.set_early_stop(pipeline, pipeline_batch)
                    return result
                if log_pipeline:
                    self.log_pipeline(pipeline)
                if self.automl.start_iteration_callback:
                    self.automl.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)
                pipeline, evaluation_result = self._compute_cv_scores(pipeline, self.automl, self.X, self.y)
                if result_callback:
                    self._add_result_callback(result_callback, pipeline, evaluation_result)
                result.add_result(pipeline, evaluation_result)
            except KeyboardInterrupt:
                terminate_early = self._handle_keyboard_interrupt()
                if terminate_early:
                    result.set_early_stop(pipeline, pipeline_batch)
                    return result
                else:
                    pipeline_batch = [pipeline] + pipeline_batch
        return result
