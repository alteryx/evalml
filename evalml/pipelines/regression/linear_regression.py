import numpy as np
import pandas as pd
from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    LinearRegressor,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler
)
from evalml.problem_types import ProblemTypes


class LinearRegressionPipeline(PipelineBase):
    """Linear Regression Pipeline for both regression"""
    name = "Linear Regressor w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    problem_types = [ProblemTypes.REGRESSION]

    hyperparameters = {
        'impute_strategy': ['most_frequent', 'mean', 'median'],
        'normalize': [False, True],
        'fit_intercept': [False, True]
    }

    def __init__(self, objective, random_state, number_features, impute_strategy, normalize=False, fit_intercept=True, n_jobs=-1):
        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        scaler = StandardScaler()

        estimator = LinearRegressor(normalize=normalize, fit_intercept=fit_intercept, n_jobs=-1)

        super().__init__(objective=objective, name=self.name, problem_type=self.problem_types, component_list=[enc, imputer, scaler, estimator])

    @property
    def feature_importances(self):
        """Return feature importances. Feature dropped by feaure selection are excluded"""
        coef_ = self.get_component("Linear Regressor")._component_obj.coef_
        importances = list(zip(self.input_feature_names, coef_))
        importances.sort(key=lambda x: -abs(x[1]))

        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df
