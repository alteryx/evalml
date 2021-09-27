"""Gaussian Mixture Clusterer."""
from sklearn.mixture import GaussianMixture as SKGaussianMixture
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class GaussianMixtureClusterer(Estimator):
    """Gaussian Mixture Clusterer.

    Args:
        n_components (int): The number of mixture components (number of clusters). Defaults to 2.
        covariance_type (str): Covariance parameters, one of {‘full’, ‘tied’, ‘diag’, ‘spherical’}. Defaults to "full".
        max_iter(int): The number of EM iterations to perform. Defaults to 100.
        n_jobs (int or None): Number of jobs to run in parallel. -1 uses all processes. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Gaussian Mixture Clusterer"
    hyperparameter_ranges = {
        "n_components": Integer(2, 10),
        "covariance_type": {"full", "tied", "diag", "spherical"},
        "max_iter": Integer(50, 300),
    }
    """{
        "n_components": Integer(2, 20),
        "covariance_type": {"full", "tied", "diag", "spherical"},
        "max_iter": Integer(50, 300),
    }"""
    model_family = ModelFamily.DISTRIBUTION
    """ModelFamily.DISTRIBUTION"""
    supported_problem_types = [
        ProblemTypes.CLUSTERING
    ]
    """[
        ProblemTypes.CLUSTERING
    ]"""

    def __init__(
        self, n_components=2, covariance_type="full", max_iter=100, n_jobs=-1, random_seed=0, **kwargs
    ):
        parameters = {
            "n_components": n_components,
            "covariance_type": covariance_type,
            "max_iter": max_iter,
        }
        parameters.update(kwargs)
        gaussian_clusterer = SKGaussianMixture(**parameters)
        super().__init__(
            parameters=parameters, component_obj=gaussian_clusterer, random_seed=random_seed
        )
