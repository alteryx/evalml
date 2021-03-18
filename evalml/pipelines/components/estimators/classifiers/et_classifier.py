from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ExtraTreesClassifier(Estimator):
    """Extra Trees Classifier."""
    name = "Extra Trees Classifier"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": Integer(4, 10)
    }
    model_family = ModelFamily.EXTRA_TREES
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    def __init__(self,
                 n_estimators=100,
                 max_features="auto",
                 max_depth=6,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 n_jobs=-1,
                 random_seed=0,
                 **kwargs):
        parameters = {"n_estimators": n_estimators,
                      "max_features": max_features,
                      "max_depth": max_depth,
                      "min_samples_split": min_samples_split,
                      "min_weight_fraction_leaf": min_weight_fraction_leaf,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        et_classifier = SKExtraTreesClassifier(random_state=random_seed,
                                               **parameters)
        super().__init__(parameters=parameters,
                         component_obj=et_classifier,
                         random_seed=random_seed)
