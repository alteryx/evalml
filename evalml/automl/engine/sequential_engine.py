from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    evaluate_pipeline,
    score_pipeline,
    train_pipeline,
)
from evalml.objectives.utils import get_objective


class SequentialComputation(EngineComputation):
    """A Future-like api for jobs created by the SequentialEngine, an Engine that sequentially computes the submitted jobs.

    In order to separate the engine from the AutoMLSearch loop, we need the sequential computations to behave the same
    way as concurrent computations from AutoMLSearch's point-of-view. One way to do this is by delaying the computation
    in the sequential engine until get_result is called. Since AutoMLSearch will call get_result only when the
    computation is "done", by always returning True in done() we make sure that get_result is called in the order that
    the jobs are submitted. So the computations happen sequentially!

    Arguments:
        work (callable): Computation that should be done by the engine.
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
        return SequentialComputation(
            work=evaluate_pipeline,
            pipeline=pipeline,
            automl_config=automl_config,
            X=X,
            y=y,
            logger=logger,
        )

    def submit_training_job(self, automl_config, pipeline, X, y):
        return SequentialComputation(
            work=train_pipeline,
            pipeline=pipeline,
            X=X,
            y=y,
            automl_config=automl_config,
            schema=False,
        )

    def submit_scoring_job(self, automl_config, pipeline, X, y, objectives):
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        computation = SequentialComputation(
            work=score_pipeline,
            pipeline=pipeline,
            X=X,
            y=y,
            objectives=objectives,
            X_schema=X.ww.schema,
            y_schema=y.ww.schema,
        )
        computation.meta_data["pipeline_name"] = pipeline.name
        return computation
