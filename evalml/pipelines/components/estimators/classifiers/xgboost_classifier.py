import pandas as pd
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import get_random_seed, import_or_raise
import re
import numpy as np
regex = re.compile(r"\[|\]|<", re.IGNORECASE)



class XGBoostClassifier(Estimator):
    """XGBoost Classifier."""
    name = "XGBoost Classifier"
    hyperparameter_ranges = {
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 10),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }
    model_family = ModelFamily.XGBOOST
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    # xgboost supports seeds from -2**31 to 2**31 - 1 inclusive. these limits ensure the random seed generated below
    # is within that range.
    SEED_MIN = -2**31
    SEED_MAX = 2**31 - 1

    def __init__(self, eta=0.1, max_depth=6, min_child_weight=1, n_estimators=100, random_state=0, **kwargs):
        random_seed = get_random_seed(random_state, self.SEED_MIN, self.SEED_MAX)
        parameters = {"eta": eta,
                      "max_depth": max_depth,
                      "min_child_weight": min_child_weight,
                      "n_estimators": n_estimators}
        parameters.update(kwargs)
        self._column_mappings = None
        xgb_error_msg = "XGBoost is not installed. Please install using `pip install xgboost.`"
        xgb = import_or_raise("xgboost", error_msg=xgb_error_msg)
        xgb_classifier = xgb.XGBClassifier(**parameters,
                                           random_state=random_seed)

        super().__init__(parameters=parameters,
                         component_obj=xgb_classifier,
                         random_state=random_state)

    def fit(self, X, y=None):
        # necessary to convert to numpy in case input DataFrame has column names that contain symbols ([, ], <) that XGBoost cannot properly handle
        if isinstance(X, pd.DataFrame):
            # old : new
            #10 : pie
            self._column_mappings = dict((col_num, col) for col_num, col in enumerate(X.columns.values))
            self.inverse = dict((v, k) for k, v in self._column_mappings.items())
            X.rename(columns=self.inverse, inplace=True)
        return super().fit(X, y)

    def predict(self, X):
        # if isinstance(X, pd.DataFrame):
        #     X.rename(columns=self.inverse, inplace=True)
        predictions = super().predict(X)
        if self._column_mappings is not None:
            predictions.rename(columns=self._column_mappings, inplace=True)
        return predictions
    
    def predict_proba(self, X):
        # if isinstance(X, pd.DataFrame):
            # X.rename(columns=self.inverse, inplace=True)
        predictions = super().predict_proba(X)
        if self._column_mappings is not None:
            predictions.rename(columns=self._column_mappings, inplace=True)
        return predictions
    
    
    @property
    def feature_importance(self):
        return self._component_obj.feature_importances_
