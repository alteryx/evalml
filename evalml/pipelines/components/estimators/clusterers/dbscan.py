"""DBSCAN Clusterer."""
from sklearn.cluster import DBSCAN as SKDBSCAN

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Clusterer
from evalml.problem_types import ProblemTypes


class DBSCANClusterer(Clusterer):
    """DBSCAN Clusterer.

    Args:
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.5.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 5.
        leaf size (int): Leaf size used when finding nearest neighbors, impacting speed and memory. Defaults to 30.
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

    def __init__(
        self, eps=0.5, min_samples=5, leaf_size=30, n_jobs=-1, random_seed=0, **kwargs
    ):
        parameters = {
            "eps": eps,
            "min_samples": min_samples,
            "leaf_size": leaf_size,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)
        dbscan_clusterer = SKDBSCAN(**parameters)
        super().__init__(
            parameters=parameters,
            component_obj=dbscan_clusterer,
            random_seed=random_seed,
        )
