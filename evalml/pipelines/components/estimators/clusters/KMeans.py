"""K Means Clusterer."""
from sklearn.cluster import KMeans as SKKMeans
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class KMeansClusterer(Estimator):
    """K Means Clusterer.

    Args:
        n_clusters (int): The number of clusters to form as well as the number of centroids to generate. Defaults to 8.
        max_iter (int): Maximum number of iterations of the k-means algorithm for a single run. Defaults to 300.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "KMeans Clusterer"
    hyperparameter_ranges = {
        "n_clusters": Integer(2, 10),
        "max_iter": Integer(50, 500),
    }
    """{
        "n_clusters": Integer(2, 10),
        "max_iter": Integer(50, 500),
    }"""
    model_family = ModelFamily.CENTROID
    """ModelFamily.CENTROID"""
    supported_problem_types = [
        ProblemTypes.CLUSTERING
    ]
    """[
        ProblemTypes.CLUSTERING
    ]"""

    def __init__(
        self, n_clusters=8, max_iter=300, n_jobs=-1, random_seed=0, **kwargs
    ):
        parameters = {
            "n_clusters": n_clusters,
            "max_iter": max_iter,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)
        kmeans_clusterer = SKKMeans(**parameters, random_state=random_seed)
        super().__init__(
            parameters=parameters, component_obj=kmeans_clusterer, random_seed=random_seed
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
