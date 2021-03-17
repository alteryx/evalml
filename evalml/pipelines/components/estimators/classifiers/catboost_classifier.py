import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    import_or_raise,
    infer_feature_types
)


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
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    def __init__(self, n_estimators=10, eta=0.03, max_depth=6, bootstrap_type=None, silent=True,
                 allow_writing_files=False, random_seed=0, **kwargs):
        parameters = {"n_estimators": n_estimators,
                      "eta": eta,
                      "max_depth": max_depth,
                      'bootstrap_type': bootstrap_type,
                      'silent': silent,
                      'allow_writing_files': allow_writing_files}
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
                         random_seed=random_seed)

    def fit(self, X, y=None):
        X = infer_feature_types(X)
        cat_cols = list(X.select('category').columns)
        self.input_feature_names = list(X.columns)
        X, y = super()._manage_woodwork(X, y)
        # For binary classification, catboost expects numeric values, so encoding before.
        if y.nunique() <= 2:
            self._label_encoder = LabelEncoder()
            y = pd.Series(self._label_encoder.fit_transform(y))
        self._component_obj.fit(X, y, silent=True, cat_features=cat_cols)
        return self

    def predict(self, X):
        X = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        predictions = self._component_obj.predict(X)
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        if self._label_encoder:
            predictions = self._label_encoder.inverse_transform(predictions.astype(np.int64))
        return infer_feature_types(predictions)

    @property
    def feature_importance(self):
        return self._component_obj.get_feature_importance()
