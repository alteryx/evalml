"""K-Nearest Neighbors Classifier."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class KNeighborsClassifier(Estimator):
    """K-Nearest Neighbors Classifier.

    Args:
        n_neighbors (int): Number of neighbors to use by default. Defaults to 5.
        weights ({‘uniform’, ‘distance’} or callable): Weight function used in prediction. Can be:

            - ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
            - ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
            - [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

            Defaults to "uniform".
        algorithm ({‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}): Algorithm used to compute the nearest neighbors:

            - ‘ball_tree’ will use BallTree
            - ‘kd_tree’ will use KDTree
            - ‘brute’ will use a brute-force search.

            ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method. Defaults to "auto".
            Note: fitting on sparse input will override the setting of this parameter, using brute force.
        leaf_size (int): Leaf size passed to BallTree or KDTree.
            This can affect the speed of the construction and query, as well as the memory required to store the tree.
            The optimal value depends on the nature of the problem. Defaults to 30.
        p (int): Power parameter for the Minkowski metric.
            When p = 1, this is equivalent to using manhattan_distance (l1),
            and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
            Defaults to 2.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "KNN Classifier"
    hyperparameter_ranges = {
        "n_neighbors": Integer(2, 12),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": Integer(10, 30),
        "p": Integer(1, 5),
    }
    """{
        "n_neighbors": Integer(2, 12),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": Integer(10, 30),
        "p": Integer(1, 5),
    }"""
    model_family = ModelFamily.K_NEIGHBORS
    """ModelFamily.K_NEIGHBORS"""
    supported_problem_types = [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""

    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "n_neighbors": n_neighbors,
            "weights": weights,
            "algorithm": algorithm,
            "leaf_size": leaf_size,
            "p": p,
        }
        parameters.update(kwargs)
        knn_classifier = SKKNeighborsClassifier(**parameters)
        super().__init__(
            parameters=parameters, component_obj=knn_classifier, random_seed=random_seed
        )

    @property
    def feature_importance(self):
        """Returns array of 0's matching the input number of features as feature_importance is not defined for KNN classifiers."""
        num_features = self._component_obj.n_features_in_
        return np.zeros(num_features)
