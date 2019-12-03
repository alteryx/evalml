from skopt.space import Real
import numpy as np

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    CatBoostClassifier,
    SimpleImputer,
    StandardScaler,
    RFClassifierSelectFromModel
)
from evalml.problem_types import ProblemTypes


class CatBoostClassificationPipeline(PipelineBase):
    """CatBoost  Pipeline for both binary and multiclass classification"""
    name = "Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    hyperparameters = {
        "impute_strategy": ["mean", "median", "most_frequent"],
    }

    def __init__(self, objective, eta, min_child_weight, max_depth, impute_strategy,
                 percent_features, number_features, n_estimators=10, n_jobs=-1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        feature_selection = RFClassifierSelectFromModel(n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        number_features=number_features,
                                                        percent_features=percent_features,
                                                        threshold=-np.inf,
                                                        n_jobs=n_jobs,
                                                        random_state=random_state)
        estimator = CatBoostClassifier()

        super().__init__(objective=objective,
                         component_list=[enc, imputer, feature_selection, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)
