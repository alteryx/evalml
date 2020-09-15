import copy
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import SEED_BOUNDS, get_random_seed, import_or_raise
from evalml.utils.gen_utils import categorical_dtypes


class CatBoostClassifier(Estimator):
    """
    CatBoost Classifier, a classifier that uses gradient-boosting on decision trees.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    """
    name = "CatBoost Classifier"
    hyperparameter_ranges = {
        "n_estimators": Integer(4, 100),
        "eta": Real(0.000001, 1),
        "max_depth": Integer(4, 10),
    }
    model_family = ModelFamily.CATBOOST
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    SEED_MIN = 0
    SEED_MAX = SEED_BOUNDS.max_bound

    def __init__(self, n_estimators=10, eta=0.03, max_depth=6, bootstrap_type=None, silent=True,
                 random_state=0, **kwargs):
        random_seed = get_random_seed(random_state, self.SEED_MIN, self.SEED_MAX)
        parameters = {"n_estimators": n_estimators,
                      "eta": eta,
                      "max_depth": max_depth,
                      'bootstrap_type': bootstrap_type,
                      'silent': silent}
        if kwargs.get('allow_writing_files', False):
            warnings.warn("Parameter allow_writing_files is being set to False in CatBoostClassifier")
        kwargs["allow_writing_files"] = False
        parameters.update(kwargs)

        cb_error_msg = "catboost is not installed. Please install using `pip install catboost.`"
        catboost = import_or_raise("catboost", error_msg=cb_error_msg)
        self._label_encoder = None
        # catboost will choose an intelligent default for bootstrap_type, so only set if provided
        cb_parameters = copy.copy(parameters)
        if bootstrap_type is None:
            cb_parameters.pop('bootstrap_type')
        cb_classifier = catboost.CatBoostClassifier(**cb_parameters,
                                                    random_seed=random_seed)
        super().__init__(parameters=parameters,
                         component_obj=cb_classifier,
                         random_state=random_state)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        cat_cols = X.select_dtypes(categorical_dtypes)

        # For binary classification, catboost expects numeric values, so encoding before.
        if y.nunique() <= 2:
            self._label_encoder = LabelEncoder()
            y = pd.Series(self._label_encoder.fit_transform(y))
        model = self._component_obj.fit(X, y, silent=True, cat_features=cat_cols)
        return model

    def predict(self, X):
        predictions = self._component_obj.predict(X)
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        if self._label_encoder:
            predictions = self._label_encoder.inverse_transform(predictions.astype(np.int64))
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions)
        return predictions

    @property
    def feature_importance(self):
        return self._component_obj.get_feature_importance()
