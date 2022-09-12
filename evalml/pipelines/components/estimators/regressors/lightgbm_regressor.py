"""LightGBM Regressor."""
import copy

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    SEED_BOUNDS,
    _rename_column_names_to_numeric,
    downcast_int_nullable_to_double,
    import_or_raise,
    infer_feature_types,
)


class LightGBMRegressor(Estimator):
    """LightGBM Regressor.

    Args:
        boosting_type (string): Type of boosting to use. Defaults to "gbdt".
            - 'gbdt' uses traditional Gradient Boosting Decision Tree
            - "dart", uses Dropouts meet Multiple Additive Regression Trees
            - "goss", uses Gradient-based One-Side Sampling
            - "rf", uses Random Forest
        learning_rate (float): Boosting learning rate. Defaults to 0.1.
        n_estimators (int): Number of boosted trees to fit. Defaults to 100.
        max_depth (int): Maximum tree depth for base learners, <=0 means no limit. Defaults to 0.
        num_leaves (int): Maximum tree leaves for base learners. Defaults to 31.
        min_child_samples (int): Minimum number of data needed in a child (leaf). Defaults to 20.
        bagging_fraction (float): LightGBM will randomly select a subset of features on each iteration (tree) without resampling if this is smaller than 1.0.
            For example, if set to 0.8, LightGBM will select 80% of features before training each tree.
            This can be used to speed up training and deal with overfitting. Defaults to 0.9.
        bagging_freq (int): Frequency for bagging. 0 means bagging is disabled.
            k means perform bagging at every k iteration.
            Every k-th iteration, LightGBM will randomly select bagging_fraction * 100 % of
            the data to use for the next k iterations. Defaults to 0.
        n_jobs (int or None): Number of threads to run in parallel. -1 uses all threads. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "LightGBM Regressor"
    hyperparameter_ranges = {
        "learning_rate": Real(0.000001, 1),
        "boosting_type": ["gbdt", "dart", "goss", "rf"],
        "n_estimators": Integer(10, 100),
        "max_depth": Integer(0, 10),
        "num_leaves": Integer(2, 100),
        "min_child_samples": Integer(1, 100),
        "bagging_fraction": Real(0.000001, 1),
        "bagging_freq": Integer(0, 1),
    }
    """{
        "learning_rate": Real(0.000001, 1),
        "boosting_type": ["gbdt", "dart", "goss", "rf"],
        "n_estimators": Integer(10, 100),
        "max_depth": Integer(0, 10),
        "num_leaves": Integer(2, 100),
        "min_child_samples": Integer(1, 100),
        "bagging_fraction": Real(0.000001, 1),
        "bagging_freq": Integer(0, 1),
    }"""
    model_family = ModelFamily.LIGHTGBM
    """ModelFamily.LIGHTGBM"""
    supported_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[ProblemTypes.REGRESSION]"""

    SEED_MIN = 0
    SEED_MAX = SEED_BOUNDS.max_bound
    """SEED_BOUNDS.max_bound"""

    def __init__(
        self,
        boosting_type="gbdt",
        learning_rate=0.1,
        n_estimators=20,
        max_depth=0,
        num_leaves=31,
        min_child_samples=20,
        bagging_fraction=0.9,
        bagging_freq=0,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "boosting_type": boosting_type,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "n_jobs": n_jobs,
            "bagging_freq": bagging_freq,
            "bagging_fraction": bagging_fraction,
        }
        parameters.update(kwargs)
        lg_parameters = copy.copy(parameters)
        # when boosting type is random forest (rf), LightGBM requires bagging_freq == 1 and  0 < bagging_fraction < 1.0
        if boosting_type == "rf":
            lg_parameters["bagging_freq"] = 1
        # when boosting type is goss, LightGBM requires bagging_fraction == 1
        elif boosting_type == "goss":
            lg_parameters["bagging_fraction"] = 1
        # avoid lightgbm warnings having to do with parameter aliases
        if (
            lg_parameters["bagging_freq"] is not None
            or lg_parameters["bagging_fraction"] is not None
        ):
            lg_parameters.update({"subsample": None, "subsample_freq": None})

        lgbm_error_msg = (
            "LightGBM is not installed. Please install using `pip install lightgbm`."
        )
        lgbm = import_or_raise("lightgbm", error_msg=lgbm_error_msg)
        self._ordinal_encoder = None

        lgbm_regressor = lgbm.sklearn.LGBMRegressor(
            random_state=random_seed, **lg_parameters
        )

        super().__init__(
            parameters=parameters,
            component_obj=lgbm_regressor,
            random_seed=random_seed,
        )

    def _encode_categories(self, X, fit=False):
        """Encodes each categorical feature using ordinal encoding."""
        X = infer_feature_types(X)
        cat_cols = list(X.ww.select("category", return_schema=True).columns)
        if fit:
            self.input_feature_names = list(X.columns)
        X_encoded = _rename_column_names_to_numeric(X)
        rename_cols_dict = dict(zip(X.columns, X_encoded.columns))
        cat_cols = [rename_cols_dict[col] for col in cat_cols]

        if len(cat_cols) == 0:
            return X_encoded
        if fit:
            self._ordinal_encoder = OrdinalEncoder()
            encoder_output = self._ordinal_encoder.fit_transform(X_encoded[cat_cols])
        else:
            encoder_output = self._ordinal_encoder.transform(X_encoded[cat_cols])
        X_encoded[cat_cols] = pd.DataFrame(encoder_output)
        X_encoded[cat_cols] = X_encoded[cat_cols].astype("category")
        return X_encoded

    def fit(self, X, y=None):
        """Fits LightGBM regressor to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self
        """
        X_encoded = self._encode_categories(X, fit=True)
        if y is not None:
            y = infer_feature_types(y)
        X_encoded = downcast_int_nullable_to_double(X_encoded)
        self._component_obj.fit(X_encoded, y)
        return self

    def predict(self, X):
        """Make predictions using fitted LightGBM regressor.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.
        """
        X_encoded = self._encode_categories(X)
        return super().predict(X_encoded)
