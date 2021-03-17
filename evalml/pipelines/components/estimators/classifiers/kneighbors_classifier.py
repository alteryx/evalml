import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class KNeighborsClassifier(Estimator):
    """
    K-Nearest Neighbors Classifier.
    """
    name = "KNN Classifier"
    hyperparameter_ranges = {
        "n_neighbors": Integer(2, 12),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": Integer(10, 30),
        "p": Integer(1, 5)
    }
    model_family = ModelFamily.K_NEIGHBORS
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    def __init__(self,
                 n_neighbors=5,
                 weights="uniform",
                 algorithm="auto",
                 leaf_size=30,
                 p=2,
                 random_seed=0,
                 **kwargs):
        parameters = {"n_neighbors": n_neighbors,
                      "weights": weights,
                      "algorithm": algorithm,
                      "leaf_size": leaf_size,
                      "p": p}
        parameters.update(kwargs)
        knn_classifier = SKKNeighborsClassifier(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=knn_classifier,
                         random_seed=random_seed)

    @property
    def feature_importance(self):
        """
        Returns array of 0's matching the input number of features as feature_importance is
        not defined for KNN classifiers.
        """
        num_features = self._component_obj.n_features_in_
        return np.zeros(num_features)
