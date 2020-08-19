# from lightgbm.sklearn import LGBMClassifier
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import get_random_seed, import_or_raise


class LightGBMClassifier(Estimator):
    """LightGBM Classifier"""
    name = "LightGBM Classifier"
    hyperparameter_ranges = {
        "learning_rate": Real(0, 1),
        "boosting_type": ["gbdt", "dart", "goss", "rf"],
        "n_estimators": Integer(10, 100)
    }
    model_family = ModelFamily.LIGHTGBM
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, boosting_type="gbdt", learning_rate=0.1, n_estimators=100, n_jobs=-1, random_state=2, **kwargs):
        parameters = {"boosting_type": boosting_type,
                      "learning_rate": learning_rate,
                      "n_estimators": n_estimators,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)

        lgbm_error_msg = "LightGBM is not installed. Please install using `pip install lightgbm`."
        lgbm = import_or_raise("lightgbm", error_msp=lgbm_error_msg)

        lgbm_classifier = lgbm.sklearn.LGBMClassifier(random_state=random_state, **parameters)

        super().__init__(parameters=parameters,
                         component_obj=lgbm_classifier,
                         random_state=random_state)
