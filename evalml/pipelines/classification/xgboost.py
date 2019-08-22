import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from evalml.pipelines import PipelineBase


class XGBoostPipeline(PipelineBase):
    name = "XGBoost w/ imputation"
    model_type = "xgboost"
    problem_type = "classification"

    hyperparameters = {
        "eta": Real(0, 1),
        "min_child_weight": Real(1, 10),
        "max_depth": Integer(1, 20),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, eta, min_child_weight, max_depth, impute_strategy, percent_features,
                 number_features, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=impute_strategy)

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
            [("imputer", imputer),
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

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

        Returns:

            self

        """
        # make everything pandas objects
        self.check_multiclass(y)
        return super().fit(X, y, objective_fit_size)

    def check_multiclass(self, y):
        num_classes = y.unique()
        if num_classes > 2:
            orig_estimator = self.pipeline.steps[-1]
            orig_params = orig_estimator.get_params()

            estimator = XGBClassifier(
                random_state=orig_params['random_state'],
                eta=orig_params['eta'],
                max_depth=orig_params['max_depth'],
                min_child_weight=orig_params['min_child_weight'],
                objective='multi:softprob',
                num_class=num_classes
            )

            self.pipeline.steps[-1] = estimator

    @property
    def feature_importances(self):
        """Return feature importances. Feature dropped by feaure selection are excluded"""
        indices = self.pipeline["feature_selection"].get_support(indices=True)
        feature_names = list(map(lambda i: self.input_feature_names[i], indices))
        importances = list(zip(feature_names, self.pipeline["estimator"].feature_importances_))  # note: this only works for binary
        importances.sort(key=lambda x: -abs(x[1]))

        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df
