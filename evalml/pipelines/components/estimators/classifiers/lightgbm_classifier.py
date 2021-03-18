import copy

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    SEED_BOUNDS,
    _convert_woodwork_types_wrapper,
    _rename_column_names_to_numeric,
    import_or_raise,
    infer_feature_types
)


class LightGBMClassifier(Estimator):
    """LightGBM Classifier"""
    name = "LightGBM Classifier"
    hyperparameter_ranges = {
        "learning_rate": Real(0.000001, 1),
        "boosting_type": ["gbdt", "dart", "goss", "rf"],
        "n_estimators": Integer(10, 100),
        "max_depth": Integer(0, 10),
        "num_leaves": Integer(2, 100),
        "min_child_samples": Integer(1, 100),
        "bagging_fraction": Real(0.000001, 1),
        "bagging_freq": Integer(0, 1)
    }
    model_family = ModelFamily.LIGHTGBM
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    SEED_MIN = 0
    SEED_MAX = SEED_BOUNDS.max_bound

    def __init__(self, boosting_type="gbdt", learning_rate=0.1, n_estimators=100, max_depth=0, num_leaves=31,
                 min_child_samples=20, n_jobs=-1, random_seed=0,
                 bagging_fraction=0.9, bagging_freq=0, **kwargs):
        parameters = {"boosting_type": boosting_type,
                      "learning_rate": learning_rate,
                      "n_estimators": n_estimators,
                      "max_depth": max_depth,
                      "num_leaves": num_leaves,
                      "min_child_samples": min_child_samples,
                      "n_jobs": n_jobs,
                      "bagging_freq": bagging_freq,
                      "bagging_fraction": bagging_fraction}
        parameters.update(kwargs)
        lg_parameters = copy.copy(parameters)
        # when boosting type is random forest (rf), LightGBM requires bagging_freq == 1 and  0 < bagging_fraction < 1.0
        if boosting_type == "rf":
            lg_parameters['bagging_freq'] = 1
        # when boosting type is goss, LightGBM requires bagging_fraction == 1
        elif boosting_type == "goss":
            lg_parameters['bagging_fraction'] = 1
        # avoid lightgbm warnings having to do with parameter aliases
        if lg_parameters['bagging_freq'] is not None or lg_parameters['bagging_fraction'] is not None:
            lg_parameters.update({'subsample': None, 'subsample_freq': None})

        lgbm_error_msg = "LightGBM is not installed. Please install using `pip install lightgbm`."
        lgbm = import_or_raise("lightgbm", error_msg=lgbm_error_msg)
        self._ordinal_encoder = None
        self._label_encoder = None

        lgbm_classifier = lgbm.sklearn.LGBMClassifier(random_state=random_seed, **lg_parameters)

        super().__init__(parameters=parameters,
                         component_obj=lgbm_classifier,
                         random_seed=random_seed)

    def _encode_categories(self, X, fit=False):
        """Encodes each categorical feature using ordinal encoding."""
        X = infer_feature_types(X)
        cat_cols = list(X.select('category').columns)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
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
        X_encoded[cat_cols] = X_encoded[cat_cols].astype('category')
        return X_encoded

    def _encode_labels(self, y):
        y_encoded = infer_feature_types(y)
        y_encoded = _convert_woodwork_types_wrapper(y_encoded.to_series())
        # change only if dtype isn't int
        if not is_integer_dtype(y_encoded):
            self._label_encoder = LabelEncoder()
            y_encoded = pd.Series(self._label_encoder.fit_transform(y_encoded), dtype='int64')
        return y_encoded

    def fit(self, X, y=None):
        X_encoded = infer_feature_types(X)
        X_encoded = self._encode_categories(X, fit=True)
        y_encoded = self._encode_labels(y)
        self._component_obj.fit(X_encoded, y_encoded)
        return self

    def predict(self, X):
        X_encoded = self._encode_categories(X)
        predictions = super().predict(X_encoded)
        if not self._label_encoder:
            return predictions
        predictions = pd.Series(self._label_encoder.inverse_transform(predictions.to_series().astype(np.int64)))
        return infer_feature_types(predictions)

    def predict_proba(self, X):
        X_encoded = self._encode_categories(X)
        return super().predict_proba(X_encoded)
