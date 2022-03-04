"""DBSCAN Clusterer."""
from sklearn.cluster import DBSCAN as SKDBSCAN

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class DBSCANClusterer(Estimator):
    """DBSCAN Clusterer.
    Args:
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.5.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 5.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "DBSCAN Clusterer"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.DENSITY
    """ModelFamily.DENSITY"""
    supported_problem_types = [ProblemTypes.CLUSTERING]
    """[
        ProblemTypes.CLUSTERING
    ]"""

    def __init__(self, eps=0.5, min_samples=5, n_jobs=-1, random_seed=0, **kwargs):
        parameters = {
            "eps": eps,
            "min_samples": min_samples,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)
        dbscan_clusterer = SKDBSCAN(**parameters)
        super().__init__(
            parameters=parameters,
            component_obj=dbscan_clusterer,
            random_seed=random_seed,
        )

    def predict(self, X=None):
        """Make predictions using selected features.
        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]. Not necessary for clustering problems.
        Returns:
            pd.Series: Predicted values.
        """
        predictions = self._component_obj.labels_
        return infer_feature_types(predictions)
