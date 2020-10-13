import copy

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import SEED_BOUNDS, get_random_seed, import_or_raise
from evalml.utils.gen_utils import categorical_dtypes


class LightGBMClassifier(Estimator):
    """LightGBM Classifier"""
    name = "LightGBM Classifier"
    hyperparameter_ranges = {
        "learning_rate": Real(0.000001, 1),
        "boosting_type": ["gbdt", "dart", "goss", "rf"],
        "n_estimators": Integer(10, 100),
        "max_depth": Integer(0, 10),
        "num_leaves": Integer(1, 100),
        "min_child_samples": Integer(1, 100)
    }
    model_family = ModelFamily.LIGHTGBM
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    SEED_MIN = 0
    SEED_MAX = SEED_BOUNDS.max_bound

    def __init__(self, boosting_type="gbdt", learning_rate=0.1, n_estimators=100, max_depth=0, num_leaves=31, min_child_samples=20, n_jobs=-1, random_state=0, bagging_fraction=0.9, bagging_freq=1, **kwargs):
        # lightGBM's current release doesn't currently support numpy.random.RandomState as the random_state value so we convert to int instead
        random_seed = get_random_seed(random_state, self.SEED_MIN, self.SEED_MAX)

        parameters = {"boosting_type": boosting_type,
                      "learning_rate": learning_rate,
                      "n_estimators": n_estimators,
                      "max_depth": max_depth,
                      "num_leaves": num_leaves,
                      "min_child_samples": min_child_samples,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)

        if boosting_type == "rf":
            parameters.update({'bagging_freq': bagging_freq, 'bagging_fraction': bagging_fraction})

        lgbm_error_msg = "LightGBM is not installed. Please install using `pip install lightgbm`."
        lgbm = import_or_raise("lightgbm", error_msg=lgbm_error_msg)
        self._ordinal_encoder = None
        self._label_encoder = None

        lgbm_classifier = lgbm.sklearn.LGBMClassifier(random_state=random_seed, **parameters)

        super().__init__(parameters=parameters,
                         component_obj=lgbm_classifier,
                         random_state=random_seed)

    def _encode_categories(self, X, fit=False):
        X2 = pd.DataFrame(copy.copy(X))
        # encode each categorical feature as an integer
        X2.columns = np.arange(len(X2.columns))
        # necessary to wipe out column names in case any names contain symbols ([, ], <) which LightGBM cannot properly handle
        cat_cols = X2.select_dtypes(categorical_dtypes).columns
        if len(cat_cols) == 0:
            return X2
        if fit:
            self._ordinal_encoder = OrdinalEncoder()
            encoder_output = self._ordinal_encoder.fit_transform(X2[cat_cols])
        else:
            encoder_output = self._ordinal_encoder.transform(X2[cat_cols])
        X2[cat_cols] = pd.DataFrame(encoder_output)
        X2[cat_cols] = X2[cat_cols].astype('category')
        return X2

    def _encode_labels(self, y):
        y1 = pd.Series(y)
        # change only if dtype isn't int
        if not is_integer_dtype(y1):
            self._label_encoder = LabelEncoder()
            y1 = pd.Series(self._label_encoder.fit_transform(y1), dtype='int64')
        return y1

    def fit(self, X, y=None):
        X2 = self._encode_categories(X, fit=True)
        y2 = self._encode_labels(y)
        return super().fit(X2, y2)

    def predict(self, X):
        X2 = self._encode_categories(X)
        predictions = super().predict(X2)
        if self._label_encoder:
            predictions = pd.Series(self._label_encoder.inverse_transform(predictions.astype(np.int64)))
        return predictions

    def predict_proba(self, X):
        X2 = self._encode_categories(X)
        return super().predict_proba(X2)
