from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    train_pipeline
)
from evalml.objectives.utils import get_objective


class SequentialComputation(EngineComputation):
    """A Future-like api for jobs created by the SequentialEngine.

    In order to separate the engine from the AutoMLSearch loop, we need the sequential computations to behave the same
    way as concurrent computations from AutoMLSearch's point-of-view. One way to do this is by delaying the computation
    in the sequential engine until get_result is called. Since AutoMLSearch will call get_result only when the
    computation is "done", by always returning True in done() we make sure that get_result is called in the order that
    the jobs are submitted. So the computations happen sequentially!
    """

    def __init__(self, work, **kwargs):
        self.work = work
        self.kwargs = kwargs
        self.meta_data = {}

    def done(self):
        """Whether the computation is done."""
        return True

    def get_result(self):
        """Gets the computation result.
        Will block until the computation is finished.

        Raises Exception: If computation fails. Returns traceback.
        """
        return self.work(**self.kwargs)

    def cancel(self):
        """Cancel the current computation."""


class SequentialEngine(EngineBase):
    """The default engine for the AutoML search. Trains and scores pipelines locally and sequentially."""

    def submit_evaluation_job(self, automl_config, pipeline, X, y):
        logger = self.setup_job_log()
        return SequentialComputation(work=evaluate_pipeline,
                                     pipeline=pipeline,
                                     automl_config=automl_config, X=X,
                                     y=y,
                                     logger=logger)

    def submit_training_job(self, automl_config, pipeline, X, y):
        return SequentialComputation(work=train_pipeline,
                                     pipeline=pipeline, X=X,
                                     y=y,
                                     optimize_thresholds=automl_config.optimize_thresholds,
                                     objective=automl_config.objective)

    def submit_scoring_job(self, automl_config, pipeline, X, y, objectives):
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        computation = SequentialComputation(work=pipeline.score,
                                            X=X, y=y, objectives=objectives)
        computation.meta_data["pipeline_name"] = pipeline.name
        return computation
