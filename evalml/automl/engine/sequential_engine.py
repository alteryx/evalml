from evalml.automl.engine import EngineBase


class SequentialEngine(EngineBase):
    """The default engine for the AutoML search. Trains and scores pipelines locally, one after another."""

    def evaluate_batch(self, pipelines):
        """Evaluate a batch of pipelines using the current dataset and AutoML state.

        Arguments:
            pipelines (list(PipelineBase)): A batch of pipelines to be fitted and evaluated.

        Returns:
            list (int): a list of the new pipeline IDs which were created by the AutoML search.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset has not been loaded into the engine.")
        new_pipeline_ids = []
        index = 0
        while self._should_continue_callback() and index < len(pipelines):
            pipeline = pipelines[index]
            self._pre_evaluation_callback(pipeline)
            evaluation_result = EngineBase.train_and_score_pipeline(pipeline, self.automl, self.X_train, self.y_train)
            new_pipeline_ids.append(self._post_evaluation_callback(pipeline, evaluation_result))
            index += 1
        return new_pipeline_ids
