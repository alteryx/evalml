import copy
import warnings

import pandas as pd
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import SEED_BOUNDS, get_random_seed, import_or_raise
from evalml.utils.gen_utils import categorical_dtypes


class CatBoostRegressor(Estimator):
    """
    CatBoost Regressor, a regressor that uses gradient-boosting on decision trees.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    """
    name = "CatBoost Regressor"
    hyperparameter_ranges = {
        "n_estimators": Integer(4, 100),
        "eta": Real(0.000001, 1),
        "max_depth": Integer(4, 10),
    }
    model_family = ModelFamily.CATBOOST
    supported_problem_types = [ProblemTypes.REGRESSION]

    SEED_MIN = 0
    SEED_MAX = SEED_BOUNDS.max_bound

    def __init__(self, n_estimators=10, eta=0.03, max_depth=6, bootstrap_type=None, silent=False,
                 random_state=0, **kwargs):
        random_seed = get_random_seed(random_state, self.SEED_MIN, self.SEED_MAX)
        parameters = {"n_estimators": n_estimators,
                      "eta": eta,
                      "max_depth": max_depth,
                      'bootstrap_type': bootstrap_type,
                      'silent': silent}
        if kwargs.get('allow_writing_files', False):
            warnings.warn("Parameter allow_writing_files is being set to False in CatBoostRegressor")
        kwargs["allow_writing_files"] = False
        parameters.update(kwargs)

        cb_error_msg = "catboost is not installed. Please install using `pip install catboost.`"
        catboost = import_or_raise("catboost", error_msg=cb_error_msg)
        # catboost will choose an intelligent default for bootstrap_type, so only set if provided
        cb_parameters = copy.copy(parameters)
        if bootstrap_type is None:
            cb_parameters.pop('bootstrap_type')
        cb_regressor = catboost.CatBoostRegressor(**cb_parameters,
                                                  random_seed=random_seed)
        super().__init__(parameters=parameters,
                         component_obj=cb_regressor,
                         random_state=random_state)

    def fit(self, X, y=None):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        cat_cols = X.select_dtypes(categorical_dtypes)
        model = self._component_obj.fit(X, y, silent=True, cat_features=cat_cols)
        return model

    @property
    def feature_importance(self):
        return self._component_obj.get_feature_importance()
