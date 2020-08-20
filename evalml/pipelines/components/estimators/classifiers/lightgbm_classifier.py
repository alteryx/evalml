import numpy as np
import pandas as pd
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise
from evalml.utils.gen_utils import get_random_seed


class LightGBMClassifier(Estimator):
    """LightGBM Classifier"""
    name = "LightGBM Classifier"
    hyperparameter_ranges = {
        "learning_rate": Real(0, 1),
        "boosting_type": ["gbdt", "dart", "goss", "rf"],
        "n_estimators": Integer(10, 100)
    }
    model_family = ModelFamily.LIGHTGBM
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, boosting_type="gbdt", learning_rate=0.1, n_estimators=100, n_jobs=-1, random_state=0, **kwargs):
        parameters = {"boosting_type": boosting_type,
                      "learning_rate": learning_rate,
                      "n_estimators": n_estimators,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)

        lgbm_error_msg = "LightGBM is not installed. Please install using `pip install lightgbm`."
        lgbm = import_or_raise("lightgbm", error_msg=lgbm_error_msg)
        
        # lightGBM's current release doesn't currently support numpy.random.RandomState as the random_state value so we convert to int instead
        rand_state = get_random_seed(random_state) if isinstance(random_state, np.random.RandomState) else random_state

        lgbm_classifier = lgbm.sklearn.LGBMClassifier(random_state=rand_state, **parameters)

        super().__init__(parameters=parameters,
                         component_obj=lgbm_classifier,
                         random_state=rand_state)

    def fit(self, X, y=None):
        # necessary to convert to numpy in case input DataFrame has column names that contain symbols ([, ], <) that LightGBM cannot properly handle
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return super().fit(X, y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return super().predict(X)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return super().predict_proba(X)
