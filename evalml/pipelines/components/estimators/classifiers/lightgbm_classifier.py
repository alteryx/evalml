import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
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

    def __init__(self, boosting_type="gbdt", learning_rate=0.1, n_estimators=100, max_depth=0, num_leaves=31, min_child_samples=20, n_jobs=-1, random_state=0, **kwargs):
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

        lgbm_error_msg = "LightGBM is not installed. Please install using `pip install lightgbm`."
        lgbm = import_or_raise("lightgbm", error_msg=lgbm_error_msg)
        self._ordinal_encoder = None

        lgbm_classifier = lgbm.sklearn.LGBMClassifier(random_state=random_seed, **parameters)

        super().__init__(parameters=parameters,
                         component_obj=lgbm_classifier,
                         random_state=random_seed)

    def _convert_to_dataframe(self, X):
        X2 = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        # rename columns in case input DataFrame has column names that contain symbols ([, ], <) that LightGBM cannot properly handle
        X2.columns = np.arange(X2.shape[1])
        cat_cols = X2.select_dtypes(categorical_dtypes).columns
        return (X2, cat_cols)

    def _make_encodings(self, X):
        X2, cat_cols = self._convert_to_dataframe(X)
        self._ordinal_encoder = OrdinalEncoder()
        # Encode the X input to be floats for all categorical components
        X2[cat_cols] = pd.DataFrame(self._ordinal_encoder.fit_transform(X2[cat_cols]))
        X2[cat_cols] = X2[cat_cols].astype('category')
        return X2

    def _encode_categories(self, X):
        X2, cat_cols = self._convert_to_dataframe(X)
        # Encode the X input to be floats for all categorical components
        X2[cat_cols] = pd.DataFrame(self._ordinal_encoder.transform(X2[cat_cols]))
        X2[cat_cols] = X2[cat_cols].astype('category')
        return X2

    def fit(self, X, y=None):
        X2 = self._make_encodings(X)
        return super().fit(X2, y)

    def predict(self, X):
        X2 = self._encode_categories(X)
        return super().predict(X2)

    def predict_proba(self, X):
        X2 = self._encode_categories(X)
        return super().predict_proba(X2)
