from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class DecisionTreeRegressor(Estimator):
    """Decision Tree Regressor."""
    name = "Decision Tree Regressor"
    hyperparameter_ranges = {
        "criterion": ["mse", "friedman_mse", "mae"],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": Integer(4, 10)
    }
    model_family = ModelFamily.DECISION_TREE
    supported_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def __init__(self,
                 criterion="mse",
                 max_features="auto",
                 max_depth=6,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 random_seed=0,
                 **kwargs):
        parameters = {"criterion": criterion,
                      "max_features": max_features,
                      "max_depth": max_depth,
                      "min_samples_split": min_samples_split,
                      "min_weight_fraction_leaf": min_weight_fraction_leaf}
        parameters.update(kwargs)
        dt_regressor = SKDecisionTreeRegressor(random_state=random_seed,
                                               **parameters)
        super().__init__(parameters=parameters,
                         component_obj=dt_regressor,
                         random_seed=random_seed)
