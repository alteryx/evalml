"""KPrototypes Clusterer."""
from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Clusterer
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, infer_feature_types


class KPrototypesClusterer(Clusterer):
    """KPrototypes Clusterer. Recommended for mixed numeric and categorical datasets.

    Args:
        n_clusters (int): The number of clusters to form as well as the number of centroids to generate. Defaults to 8.
        max_iter (int): Maximum number of iterations of the k-prototypes algorithm for a single run. Defaults to 300.
        n_init (int): Number of time the algorithm will be run with different centroid seeds. Defaults to 10.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "KPrototypes Clusterer"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.CENTROID
    """ModelFamily.CENTROID"""
    supported_problem_types = [ProblemTypes.CLUSTERING]
    """[
        ProblemTypes.CLUSTERING
    ]"""

    def __init__(
        self, n_clusters=8, max_iter=300, n_init=10, n_jobs=-1, random_seed=0, **kwargs
    ):
        parameters = {
            "n_clusters": n_clusters,
            "max_iter": max_iter,
            "n_init": n_init,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)
        kprototypes_error_msg = "KModes is not installed. Please install using `pip install kmodes` to run KPrototypes."
        kmodes = import_or_raise("kmodes.kprototypes", error_msg=kprototypes_error_msg)
        kprototypes_clusterer = kmodes.KPrototypes(
            **parameters, random_state=random_seed
        )
        super().__init__(
            parameters=parameters,
            component_obj=kprototypes_clusterer,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits KPrototypes clusterer to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples]. Not necessary and ignored for clustering problems.

        Returns:
            self
        """
        X = infer_feature_types(X)
        cat_col_names = list(X.ww.select("category", return_schema=True).columns)
        cat_col_idxs = [
            col_idx
            for col_idx, col_name in enumerate(X.columns)
            if col_name in cat_col_names
        ]
        self._component_obj.fit(X, categorical=cat_col_idxs)
        return self
