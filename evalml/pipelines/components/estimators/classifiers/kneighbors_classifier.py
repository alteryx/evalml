import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as SKRandomForesClassifier
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class KNeighborsClassifier(Estimator):
    """
    K-Nearest Neighbors Classifier.
    """
    name = "K-Nearest Neighbors Classifier"
    hyperparameter_ranges = {
        "n_neighbors": Integer(2, 12),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": Integer(10, 30),
        "p": Integer(1, 5)
    }
    model_family = ModelFamily.K_NEIGHBORS
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self,
                 n_neighbors=5,
                 weights="uniform",
                 algorithm="auto",
                 leaf_size=30,
                 p=2,
                 random_state=0, #Capture random_state so it doesn't get into params
                 **kwargs):
        parameters = {"n_neighbors": n_neighbors,
                      "weights": weights,
                      "algorithm": algorithm,
                      "leaf_size": leaf_size,
                      "p": p}
        parameters.update(kwargs)

        knn_classifier = SKKNeighborsClassifier(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=knn_classifier)

    @property
    def feature_importance(self):
        """
        Restrictions: return must be a numpy array and the length must match the input num_cols
        """
        num_features = self._component_obj.n_features_in_
        return np.zeros(num_features)