"""Agglomerative Clusterer."""
from sklearn.cluster import AgglomerativeClustering as SKAgglomerativeClustering
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class AgglomerativeClusterer(Estimator):
    """Agglomerative Clusterer.

    Args:
        n_clusters (float): The number of clusters to find. Defaults to 2.
        linkage (int): Which linkage criterion to use. Defaults to "ward".
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Agglomerative Clusterer"
    hyperparameter_ranges = {
        "n_clusters": Integer(2, 10),
        "linkage": {"ward", "complete", "average", "single"},
    }
    """{
        "n_clusters": Integer(2, 20),
        "linkage": {"ward", "complete", "average", "single"},
    }"""
    model_family = ModelFamily.HIERARCHY
    """ModelFamily.HIERARCHY"""
    supported_problem_types = [
        ProblemTypes.CLUSTERING
    ]
    """[
        ProblemTypes.CLUSTERING
    ]"""

    def __init__(
        self, n_clusters=2, linkage="ward", n_jobs=-1, random_seed=0, **kwargs
    ):
        parameters = {
            "n_clusters": n_clusters,
            "linkage": linkage,
        }
        parameters.update(kwargs)
        agglo_clusterer = SKAgglomerativeClustering(**parameters)
        super().__init__(
            parameters=parameters, component_obj=agglo_clusterer, random_seed=random_seed
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
