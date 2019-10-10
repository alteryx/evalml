import warnings

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from skopt.space import Integer, Real


class XGBoostPipeline(PipelineBase):
    """XGBoost Pipeline for both binary and multiclass classification"""
    name = "XGBoost w/ imputation"
    model_type = ModelTypes.XGBOOST
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
        imputer = SimpleImputer(strategy=impute_strategy)
        enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)

        estimator = XGBClassifier(
            random_state=random_state,
            eta=eta,
            max_depth=max_depth,
            min_child_weight=min_child_weight
        )

        feature_selection = SelectFromModel(
            estimator=estimator,
            max_features=max(1, int(percent_features * number_features)),
            threshold=-np.inf
        )

        self.pipeline = Pipeline(
            [("encoder", enc),
             ("imputer", imputer),
             ("feature_selection", feature_selection),
             ("estimator", estimator)]
        )

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
            params = self.pipeline['estimator'].get_params()
            params.update(
                {
                    "objective": 'multi:softprob',
                    "num_class": num_classes
                })

            estimator = XGBClassifier(**params)
            self.pipeline.steps[-1] = ('estimator', estimator)

        return super().fit(X, y, objective_fit_size)

    @property
    def feature_importances(self):
        """Return feature importances. Feature dropped by feaure selection are excluded"""
        indices = self.pipeline["feature_selection"].get_support(indices=True)
        feature_names = list(map(lambda i: self.input_feature_names[i], indices))
        importances = list(zip(feature_names, self.pipeline["estimator"].feature_importances_))
        importances.sort(key=lambda x: -abs(x[1]))

        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df
