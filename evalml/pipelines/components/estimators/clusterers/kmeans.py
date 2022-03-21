"""KMeans Clusterer."""
from sklearn.cluster import KMeans as SKKMeans

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Unsupervised
from evalml.problem_types import ProblemTypes


class KMeansClusterer(Unsupervised):
    """KMeans Clusterer. Recommended for numeric-only datasets.

    Args:
        n_clusters (int): The number of clusters to form as well as the number of centroids to generate. Defaults to 8.
        max_iter (int): Maximum number of iterations of the k-means algorithm for a single run. Defaults to 300.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "KMeans Clusterer"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.CENTROID
    """ModelFamily.CENTROID"""
    supported_problem_types = [ProblemTypes.CLUSTERING]
    """[
        ProblemTypes.CLUSTERING
    ]"""

    def __init__(self, n_clusters=8, max_iter=300, random_seed=0, **kwargs):
        parameters = {
            "n_clusters": n_clusters,
            "max_iter": max_iter,
        }
        parameters.update(kwargs)
        kmeans_clusterer = SKKMeans(**parameters, random_state=random_seed)
        super().__init__(
            parameters=parameters,
            component_obj=kmeans_clusterer,
            random_seed=random_seed,
        )
