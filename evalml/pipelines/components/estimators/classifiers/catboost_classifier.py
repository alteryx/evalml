import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise

class CatBoostClassifier(Estimator):
    """
    CatBoost Classifier
    """
    name = "CatBoost Classifier"
    component_type = ComponentTypes.CLASSIFIER
    _needs_fitting = True
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 16)
    }
    model_type = ModelTypes.CATBOOST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, n_estimators=1000, eta=0.03, max_depth=6, random_state=0):
        parameters = {"n_estimators": n_estimators,
                      "eta": eta,
                      "max_depth": max_depth}

        cb_error_msg = "catboost is not installed. Please install using `pip install catboost.`"
        catboost = import_or_raise("catboost", error_msg=cb_error_msg)
        self._label_encoder = None
        cb_classifier = catboost.CatBoostClassifier(n_estimators=n_estimators,
                                                    eta=eta,
                                                    max_depth=max_depth,
                                                    silent=True,
                                                    random_state=random_state,
                                                    allow_writing_files=False)
        super().__init__(parameters=parameters,
                         component_obj=cb_classifier,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self
        """
        cat_cols = X.select_dtypes(['object', 'category'])

        # For binary classification, catboost expects numeric values, so encoding before.
        if y.nunique() <= 2:
            self._label_encoder = LabelEncoder()
            y = pd.Series(self._label_encoder.fit_transform(y))
        model = self._component_obj.fit(X, y, silent=True, cat_features=cat_cols)
        shutil.rmtree('catboost_info', ignore_errors=True)
        return model

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (DataFrame) : features

        Returns:
            Series : estimated labels
        """
        predictions = self._component_obj.predict(X)
        if self._label_encoder:
            return self._label_encoder.inverse_transform(predictions.astype(np.int64))

        return predictions

    @property
    def feature_importances(self):
        return self._component_obj.get_feature_importance()
