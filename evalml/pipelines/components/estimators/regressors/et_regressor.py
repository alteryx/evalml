"""Extra Trees Regressor."""
from sklearn.ensemble import ExtraTreesRegressor as SKExtraTreesRegressor
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ExtraTreesRegressor(Estimator):
    """Extra Trees Regressor.

    Args:
        n_estimators (float): The number of trees in the forest. Defaults to 100.
        max_features (int, float or {"auto", "sqrt", "log2"}): The number of features to consider when looking for the best split:

            - If int, then consider max_features features at each split.
            - If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
            - If "auto", then max_features=sqrt(n_features).
            - If "sqrt", then max_features=sqrt(n_features).
            - If "log2", then max_features=log2(n_features).
            - If None, then max_features = n_features.

            The search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
            Defaults to "auto".
        max_depth (int): The maximum depth of the tree. Defaults to 6.
        min_samples_split (int or float): The minimum number of samples required to split an internal node:

            - If int, then consider min_samples_split as the minimum number.
            - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

        Defaults to 2.
        min_weight_fraction_leaf (float): The minimum weighted fraction of the sum total of weights
            (of all the input samples) required to be at a leaf node. Defaults to 0.0.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Extra Trees Regressor"
    hyperparameter_ranges = {
        "n_estimators": Integer(10, 1000),
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": Integer(4, 10),
    }
    """{
        "n_estimators": Integer(10, 1000),
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": Integer(4, 10),
    }"""
    model_family = ModelFamily.EXTRA_TREES
    """ModelFamily.EXTRA_TREES"""
    supported_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""

    def __init__(
        self,
        n_estimators=100,
        max_features="auto",
        max_depth=6,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        et_regressor = SKExtraTreesRegressor(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters,
            component_obj=et_regressor,
            random_seed=random_seed,
        )
