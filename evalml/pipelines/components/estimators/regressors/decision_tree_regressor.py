"""Decision Tree Regressor."""
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class DecisionTreeRegressor(Estimator):
    """Decision Tree Regressor.

    Args:
        criterion ({"mse", "friedman_mse", "mae", "poisson"}): The function to measure the quality of a split.
            Supported criteria are:

                - "mse" for the mean squared error, which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node
                - "friedman_mse", which uses mean squared error with Friedman"s improvement score for potential splits
                - "mae" for the mean absolute error, which minimizes the L1 loss using the median of each terminal node,
                - "poisson" which uses reduction in Poisson deviance to find splits.
        max_features (int, float or {"auto", "sqrt", "log2"}): The number of features to consider when looking for the best split:

            - If int, then consider max_features features at each split.
            - If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
            - If "auto", then max_features=sqrt(n_features).
            - If "sqrt", then max_features=sqrt(n_features).
            - If "log2", then max_features=log2(n_features).
            - If None, then max_features = n_features.

            The search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
        max_depth (int): The maximum depth of the tree. Defaults to 6.
        min_samples_split (int or float): The minimum number of samples required to split an internal node:

            - If int, then consider min_samples_split as the minimum number.
            - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

            Defaults to 2.
        min_weight_fraction_leaf (float): The minimum weighted fraction of the sum total of weights
            (of all the input samples) required to be at a leaf node. Defaults to 0.0.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Decision Tree Regressor"
    hyperparameter_ranges = {
        "criterion": ["mse", "friedman_mse", "mae"],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": Integer(4, 10),
    }
    """{
        "criterion": ["mse", "friedman_mse", "mae"],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": Integer(4, 10),
    }"""
    model_family = ModelFamily.DECISION_TREE
    """ModelFamily.DECISION_TREE"""
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
        criterion="mse",
        max_features="auto",
        max_depth=6,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "criterion": criterion,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
        }
        parameters.update(kwargs)
        dt_regressor = SKDecisionTreeRegressor(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters,
            component_obj=dt_regressor,
            random_seed=random_seed,
        )
