"""EvalML Engine classes used to evaluate pipelines in AutoMLSearch."""
from evalml.automl.engine.engine_base import (
    EngineBase,
    EngineComputation,
    train_pipeline,
    train_and_score_pipeline,
    evaluate_pipeline,
)
from evalml.automl.engine.sequential_engine import SequentialEngine
from evalml.automl.engine.dask_engine import DaskEngine
from evalml.automl.engine.cf_engine import CFEngine
