import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise


class CatBoostClassifier(Estimator):
    """
    CatBoost Classifier, a classifier that uses gradient-boosting on decision trees.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    """
    name = "CatBoost Classifier"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 16),
    }
    model_type = ModelTypes.CATBOOST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, n_estimators=1000, eta=0.03, max_depth=6, bootstrap_type=None, random_state=0):
        parameters = {"n_estimators": n_estimators,
                      "eta": eta,
                      "max_depth": max_depth}
        if bootstrap_type is not None:
            parameters['bootstrap_type'] = bootstrap_type

        cb_error_msg = "catboost is not installed. Please install using `pip install catboost.`"
        catboost = import_or_raise("catboost", error_msg=cb_error_msg)
        self._label_encoder = None
        cb_classifier = catboost.CatBoostClassifier(**parameters,
                                                    silent=True,
                                                    allow_writing_files=False)
        super().__init__(parameters=parameters,
                         component_obj=cb_classifier,
                         random_state=random_state)

    def fit(self, X, y=None):
        cat_cols = X.select_dtypes(['category', 'object'])

        # For binary classification, catboost expects numeric values, so encoding before.
        if y.nunique() <= 2:
            self._label_encoder = LabelEncoder()
            y = pd.Series(self._label_encoder.fit_transform(y))
        model = self._component_obj.fit(X, y, silent=True, cat_features=cat_cols)
        return model

    def predict(self, X):
        predictions = self._component_obj.predict(X)
        if self._label_encoder:
            return self._label_encoder.inverse_transform(predictions.astype(np.int64))

        return predictions

    @property
    def feature_importances(self):
        return self._component_obj.get_feature_importance()
