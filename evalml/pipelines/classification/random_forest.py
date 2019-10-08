import numpy as np
import pandas as pd
from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    OneHotEncoder,
    RandomForestClassifier,
    SelectFromModel,
    SimpleImputer
)
from evalml.problem_types import ProblemTypes


class RFClassificationPipeline(PipelineBase):
    """Random Forest Pipeline for both binary and multiclass classification"""
    name = "Random Forest w/ imputation"
    model_type = ModelTypes.RANDOM_FOREST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, n_estimators, max_depth, impute_strategy,
                 percent_features, number_features, n_jobs=1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        estimator = RandomForestClassifier(random_state=random_state,
                                           n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           n_jobs=n_jobs)

        feature_selection = SelectFromModel(
            estimator=estimator._component_obj,
            number_features=number_features,
            percent_features=percent_features,
            threshold=-np.inf
        )

        super().__init__(objective=objective, name=self.name, problem_type=self.problem_types, component_list=[enc, imputer, feature_selection, estimator])

    @property
    def feature_importances(self):
        """Return feature importances. Feature dropped by feaure selection are excluded"""
        indices = self.get_component('Select From Model')._component_obj.get_support(indices=True)
        feature_names = list(map(lambda i: self.input_feature_names[i], indices))
        importances = list(zip(feature_names, self.get_component("Random Forest Classifier")._component_obj.feature_importances_))
        importances.sort(key=lambda x: -abs(x[1]))

        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df
