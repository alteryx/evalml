"""EvalML Engine classes used to evaluate pipelines in AutoMLSearch."""
from .engine_base import (
    EngineBase,
    EngineComputation,
    train_pipeline,
    train_and_score_pipeline,
    evaluate_pipeline,
)
from .sequential_engine import SequentialEngine
from .dask_engine import DaskEngine
from .cf_engine import CFEngine
