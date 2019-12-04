from skopt.space import Integer, Real
import numpy as np

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    CatBoostClassifier,
    SimpleImputer,
    StandardScaler,
)
from evalml.problem_types import ProblemTypes


class CatBoostClassificationPipeline(PipelineBase):
    """CatBoost Pipeline for both binary and multiclass classification"""
    name = "CatBoost Classifier w/ Simple Imputer"
    model_type = ModelTypes.CATBOOST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
    hyperparameters = {
        "impute_strategy": ["constant", "most_frequent"],
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
    }

    def __init__(self, objective, impute_strategy, n_estimators, eta, max_depth, n_jobs=-1, random_state=0):
        # note: impute_strategy must support both string and numeric data
        imputer = SimpleImputer(impute_strategy=impute_strategy)
        estimator = CatBoostClassifier(n_estimators=n_estimators,
                                       eta=eta, max_depth=max_depth)

        super().__init__(objective=objective,
                         component_list=[imputer, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)
