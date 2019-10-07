import numpy as np
import pandas as pd

from skopt.space import Integer, Real
from evalml.pipelines import Pipeline, PipelineBase
from evalml.pipelines.components import (
    OneHotEncoder,
    SelectFromModel,
    SimpleImputer,
    XGBoostClassifier
)
from evalml.problem_types import ProblemTypes


class XGBoostPipeline(PipelineBase):
    """XGBoost Pipeline for both binary and multiclass classification"""
    name = "XGBoost w/ imputation"
    model_type = "xgboost"
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    hyperparameters = {
        "eta": Real(0, 1),
        "min_child_weight": Real(1, 10),
        "max_depth": Integer(1, 20),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, eta, min_child_weight, max_depth, impute_strategy,
                 percent_features, number_features, n_jobs=1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        estimator = XGBoostClassifier(
            random_state=random_state,
            eta=eta,
            max_depth=max_depth,
            min_child_weight=min_child_weight
        )
        feature_selection = SelectFromModel(
            estimator=estimator._component_obj,
            number_features=number_features,
            percent_features=percent_features,
            threshold=-np.inf
        )
        self.pipeline = Pipeline(objective=objective, name="", problem_type=None, component_list=[enc, imputer, feature_selection, estimator])

        super().__init__(objective=objective, random_state=random_state)

    # Need to override fit for multiclass
    def fit(self, X, y, objective_fit_size=.2):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

        Returns:

            self

        """
        # check if problem is multiclass
        num_classes = len(np.unique(y))
        if num_classes > 2:
            params = self.pipeline.get_component('XGBoost Classifier')._component_obj.get_params()
            # params = self.pipeline['estimator'].get_params()
            params.update(
                {
                    "objective": 'multi:softprob',
                    "num_class": num_classes
                })

            estimator = XGBoostClassifier(**params)
            self.pipeline.component_list[-1] = estimator

        return super().fit(X, y, objective_fit_size)

    @property
    def feature_importances(self):
        """Return feature importances. Feature dropped by feaure selection are excluded"""
        indices = self.pipeline.get_component('Select From Model')._component_obj.get_support(indices=True)
        feature_names = list(map(lambda i: self.input_feature_names[i], indices))
        importances = list(zip(feature_names, self.pipeline.get_component("XGBoost Classifier")._component_obj.feature_importances_))
        importances.sort(key=lambda x: -abs(x[1]))

        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df
