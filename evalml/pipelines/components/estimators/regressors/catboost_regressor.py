import shutil

from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise


class CatBoostRegressor(Estimator):
    """
    CatBoost Regressor
    """
    name = "CatBoost Regressor"
    component_type = ComponentTypes.REGRESSOR
    _needs_fitting = True
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 16)
    }
    model_type = ModelTypes.CATBOOST
    problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, n_estimators=1000, eta=0.03, max_depth=6, random_state=0):
        parameters = {"n_estimators": n_estimators,
                      "eta": eta,
                      "max_depth": max_depth}

        cb_error_msg = "catboost is not installed. Please install using `pip install catboost.`"
        catboost = import_or_raise("catboost", error_msg=cb_error_msg)
        cb_regressor = catboost.CatBoostRegressor(n_estimators=n_estimators,
                                                  eta=eta,
                                                  max_depth=max_depth,
                                                  silent=True,
                                                  random_state=random_state,
                                                  allow_writing_files=False)
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
        cat_cols = X.select_dtypes(['object', 'category'])
        model = self._component_obj.fit(X, y, silent=True, cat_features=cat_cols)
        shutil.rmtree('catboost_info', ignore_errors=True)
        return model

    @property
    def feature_importances(self):
        return self._component_obj.get_feature_importance()
