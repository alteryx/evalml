"""KModes Clusterer."""
from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, infer_feature_types


class KModesClusterer(Estimator):
    """KModes Clusterer. Recommended for categorical-only datasets.

    Args:
        n_clusters (int): The number of clusters to form as well as the number of centroids to generate. Defaults to 8.
        max_iter (int): Maximum number of iterations of the k-modes algorithm for a single run. Defaults to 300.
        n_init (int): Number of time the algorithm will be run with different centroid seeds. Defaults to 10.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "KModes Clusterer"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.CENTROID
    """ModelFamily.CENTROID"""
    supported_problem_types = [ProblemTypes.CLUSTERING]
    """[
        ProblemTypes.CLUSTERING
    ]"""

    def __init__(self, n_clusters=8, max_iter=300, n_init=10, n_jobs=-1, random_seed=0, **kwargs):
        parameters = {
            "n_clusters": n_clusters,
            "max_iter": max_iter,
            "n_init": n_init,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)
        kmodes_error_msg = (
            "KModes is not installed. Please install using `pip install kmodes.`"
        )
        kmodes = import_or_raise("kmodes.kmodes", error_msg=kmodes_error_msg)
        kmodes_clusterer = kmodes.KModes(**parameters, random_state=random_seed)
        super().__init__(
            parameters=parameters,
            component_obj=kmodes_clusterer,
            random_seed=random_seed,
        )

    def predict(self, X=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]. Not used for clustering problems.

        Returns:
            pd.Series: Predicted values.
        """
        predictions = self._component_obj.labels_
        return infer_feature_types(predictions)
