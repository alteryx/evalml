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
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    _rename_column_names_to_numeric
)


class GAMRegressor(Estimator):
    """GAM Regressor"""
    name = "GAM Regressor"
    hyperparameter_ranges = {
        "solver": ["IRLSM", "L_BFGS"],
        "alpha": Real(0.000001, 1),
        "lambda": Real(0.000001, 1)
    }
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    SEED_MIN = 0
    SEED_MAX = SEED_BOUNDS.max_bound

    def __init__(self, family='AUTO', solver="AUTO", stopping_metric="logloss", keep_cross_validation_models=False, random_state=0, **kwargs):
        random_seed = get_random_seed(random_state, self.SEED_MIN, self.SEED_MAX)

        self.parameters = {"family": family,
                           "solver": solver,
                           "stopping_metric": stopping_metric,
                           "keep_cross_validation_models": keep_cross_validation_models,
                           "seed": random_seed}
        self.parameters.update(kwargs)

        h2o_error_msg = "H2O is not installed. please install using `pip install h2o`."
        self.h2o = import_or_raise("h2o", error_msg=h2o_error_msg)
        self.h2o.init()

        self.h2o_model = self.h2o.estimators.gam.H2OGeneralizedAdditiveEstimator

        super().__init__(parameters=self.parameters,
                         random_state=random_seed)

    def _update_params(self, X, y):
        feat_columns = list(X.columns)
        new_params = {'gam_columns': feat_columns,
                      "lambda_search": True}
        if y.nunique() == 3:
            new_params.update({"family": "multinomial",
                               "link": "Family_Default"})
        elif y.nunique() > 3:
            new_params.update({"family": "ordinal",
                              "solver": "GRADIENT_DESCENT_LH",
                               "link": "Family_Default"})
        else:
            new_params.update({"family": "binomial",
                               "link": "Logit"})

        return new_params

    def train(self, X, y=None):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        if y is not None:
            y = _convert_to_woodwork_structure(y)
            y = _convert_woodwork_types_wrapper(y.to_series())
        try:
            training_frame = X.merge(y, left_index=True, right_index=True)
            training_frame = self.h2o.H2OFrame(training_frame)
            training_frame[y.name] = training_frame[y.name].asfactor()

            new_params = self._update_params(X, y)
            self.parameters.update(new_params)
            self.h2o_model(**self.parameters)

            self.h2o_model.train(x=list(X.columns), y=y.name, training_frame=training_frame)
            return self.h2o_model
        except AttributeError:
            raise MethodPropertyNotFoundError("Component requires a train method or a component_obj that implements train")

    def predict(self, X):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        X = self.h2o.H2OFrame(X)
        predictions = self.h2o_model.predict(X)
        predictions = predictions.as_data_frame(use_pandas=True)
        return predictions

    @property
    def feature_importance(self):
        return self.h2o_model.varimp()