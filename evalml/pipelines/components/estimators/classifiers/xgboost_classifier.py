"""XGBoost Classifier."""
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    _rename_column_names_to_numeric,
    import_or_raise,
    infer_feature_types,
)


class XGBoostClassifier(Estimator):
    """XGBoost Classifier.

    Args:
        eta (float): Boosting learning rate. Defaults to 0.1.
        max_depth (int): Maximum tree depth for base learners. Defaults to 6.
        min_child_weight (float): Minimum sum of instance weight (hessian) needed in a child. Defaults to 1.0
        n_estimators (int): Number of gradient boosted trees. Equivalent to number of boosting rounds. Defaults to 100.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        n_jobs (int): Number of parallel threads used to run xgboost. Note that creating thread contention will significantly slow down the algorithm. Defaults to 12.
    """

    name = "XGBoost Classifier"
    hyperparameter_ranges = {
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 10),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }
    """{
        "eta": Real(0.000001, 1),
        "max_depth": Integer(1, 10),
        "min_child_weight": Real(1, 10),
        "n_estimators": Integer(1, 1000),
    }"""
    model_family = ModelFamily.XGBOOST
    """ModelFamily.XGBOOST"""
    supported_problem_types = [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""

    # xgboost supports seeds from -2**31 to 2**31 - 1 inclusive. these limits ensure the random seed generated below
    # is within that range.
    SEED_MIN = -(2 ** 31)
    SEED_MAX = 2 ** 31 - 1

    def __init__(
        self,
        eta=0.1,
        max_depth=6,
        min_child_weight=1,
        n_estimators=100,
        random_seed=0,
        eval_metric="logloss",
        n_jobs=12,
        **kwargs,
    ):
        parameters = {
            "eta": eta,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
            "eval_metric": eval_metric,
        }
        parameters.update(kwargs)
        if "use_label_encoder" in parameters:
            parameters.pop("use_label_encoder")
        xgb_error_msg = (
            "XGBoost is not installed. Please install using `pip install xgboost.`"
        )
        xgb = import_or_raise("xgboost", error_msg=xgb_error_msg)
        xgb_classifier = xgb.XGBClassifier(
            use_label_encoder=False, random_state=random_seed, **parameters
        )
        self._label_encoder = None
        super().__init__(
            parameters=parameters, component_obj=xgb_classifier, random_seed=random_seed
        )

    @staticmethod
    def _convert_bool_to_int(X):
        return {
            col: "Integer" for col in X.ww.select("boolean", return_schema=True).columns
        }

    def _label_encode(self, y):
        if not is_integer_dtype(y):
            self._label_encoder = LabelEncoder()
            y = pd.Series(self._label_encoder.fit_transform(y), dtype="int64")
        return y

    def fit(self, X, y=None):
        """Fits XGBoost classifier component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self
        """
        X, y = super()._manage_woodwork(X, y)
        X.ww.set_types(self._convert_bool_to_int(X))
        self.input_feature_names = list(X.columns)
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        y = self._label_encode(y)
        self._component_obj.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions using the fitted XGBoost classifier.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.DataFrame: Predicted values.
        """
        X, _ = super()._manage_woodwork(X)
        X.ww.set_types(self._convert_bool_to_int(X))
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        predictions = super().predict(X)
        if not self._label_encoder:
            return predictions
        predictions = pd.Series(
            self._label_encoder.inverse_transform(predictions.astype(np.int64))
        )
        return infer_feature_types(predictions)

    def predict_proba(self, X):
        """Make predictions using the fitted CatBoost classifier.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.DataFrame: Predicted values.
        """
        X, _ = super()._manage_woodwork(X)
        X.ww.set_types(self._convert_bool_to_int(X))
        X = _rename_column_names_to_numeric(X, flatten_tuples=False)
        return super().predict_proba(X)

    @property
    def feature_importance(self):
        """Feature importance of fitted XGBoost classifier."""
        return self._component_obj.feature_importances_
