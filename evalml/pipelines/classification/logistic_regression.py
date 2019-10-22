import numpy as np
import pandas as pd
from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    LogisticRegressionClassifier,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler
)
from evalml.problem_types import ProblemTypes


class LogisticRegressionPipeline(PipelineBase):
    """Logistic Regression Pipeline for both binary and multiclass classification"""
    name = "Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }

    def __init__(self, objective, penalty, C, impute_strategy,
                 number_features, n_jobs=-1, random_state=0):
        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        scaler = StandardScaler()

        estimator = LogisticRegressionClassifier(random_state=random_state,
                                                 penalty=penalty,
                                                 C=C,
                                                 n_jobs=-1)

        super().__init__(objective=objective, name=self.name, problem_type=self.problem_types, component_list=[enc, imputer, scaler, estimator])

    # @property
    # def feature_importances(self):
    #     """Return feature importances. Feature dropped by feaure selection are excluded"""
    #     coef_ = self.get_component("Logistic Regression Classifier")._component_obj.coef_

    #     # binary classification case
    #     if len(coef_) <= 2:
    #         importances = list(zip(self.input_feature_names, coef_[0]))
    #         importances.sort(key=lambda x: -abs(x[1]))
    #     else:
    #         # mutliclass classification case
    #         importances = list(zip(self.input_feature_names, np.linalg.norm(coef_, axis=0, ord=2)))
    #         importances.sort(key=lambda x: -(x[1]))

    #     df = pd.DataFrame(importances, columns=["feature", "importance"])
    #     return df
