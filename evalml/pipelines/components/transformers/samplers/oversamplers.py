from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseOverSampler,
)
from evalml.utils.woodwork_utils import infer_feature_types


class SMOTESampler(BaseOverSampler):
    """SMOTE Oversampler component. Works on numerical datasets only. This component is only run during training and not during predict.

    Arguments:
        sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
            of the minority to majority class after oversampling. We will create the a sampling dictionary using this ratio, with the keys corresponding to the class
            and the values responding to the number of samples. Defaults to 0.25.
        k_neighbors_default (int): The number of nearest neighbors used to construct synthetic samples. This is the default value used, but the actual k_neighbors value might be smaller
            if there are less samples. Defaults to 5.
        n_jobs (int): The number of CPU cores to use. Defaults to -1.
        random_seed (int): The seed to use for random sampling. Defaults to 0.
    """

    name = "SMOTE Oversampler"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        sampling_ratio=0.25,
        k_neighbors_default=5,
        n_jobs=-1,
        random_seed=0,
        **kwargs
    ):
        super().__init__(
            "SMOTE",
            sampling_ratio=sampling_ratio,
            k_neighbors_default=k_neighbors_default,
            n_jobs=n_jobs,
            random_seed=random_seed,
            **kwargs
        )


class SMOTENCSampler(BaseOverSampler):
    """SMOTENC Oversampler component. Uses SMOTENC to generate synthetic samples. Works on a mix of nomerical and categorical columns.
    Input data must be Woodwork type, and this component is only run during training and not during predict.

    Arguments:
        sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
            of the minority to majority class after oversampling. We will create the a sampling dictionary using this ratio, with the keys corresponding to the class
            and the values responding to the number of samples. Defaults to 0.25.
        k_neighbors_default (int): The number of nearest neighbors used to construct synthetic samples. This is the default value used, but the actual k_neighbors value might be smaller
            if there are less samples. Defaults to 5.
        n_jobs (int): The number of CPU cores to use. Defaults to -1.
        random_seed (int): The seed to use for random sampling. Defaults to 0.
    """

    name = "SMOTENC Oversampler"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        sampling_ratio=0.25,
        k_neighbors_default=5,
        n_jobs=-1,
        random_seed=0,
        **kwargs
    ):
        self.categorical_features = None
        super().__init__(
            "SMOTENC",
            sampling_ratio=sampling_ratio,
            k_neighbors_default=k_neighbors_default,
            n_jobs=n_jobs,
            random_seed=random_seed,
            **kwargs
        )

    def _get_categorical(self, X):
        X = infer_feature_types(X)
        self.categorical_features = [
            i
            for i, val in enumerate(X.ww.types["Logical Type"].items())
            if str(val[1]) in {"Boolean", "Categorical"}
        ]
        self._parameters["categorical_features"] = self.categorical_features

    def fit(self, X, y):
        # get categorical features first
        self._get_categorical(X)
        super().fit(X, y)


class SMOTENSampler(BaseOverSampler):
    """
    SMOTEN Oversampler component. Uses SMOTEN to generate synthetic samples. Works for purely categorical datasets.
    This component is only run during training and not during predict.

    Arguments:
        sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
            of the minority to majority class after oversampling. We will create the a sampling dictionary using this ratio, with the keys corresponding to the class
            and the values responding to the number of samples. Defaults to 0.25.
        k_neighbors_default (int): The number of nearest neighbors used to construct synthetic samples. This is the default value used, but the actual k_neighbors value might be smaller
            if there are less samples. Defaults to 5.
        n_jobs (int): The number of CPU cores to use. Defaults to -1.
        random_seed (int): The seed to use for random sampling. Defaults to 0.
    """

    name = "SMOTEN Oversampler"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        sampling_ratio=0.25,
        k_neighbors_default=5,
        n_jobs=-1,
        random_seed=0,
        **kwargs
    ):
        super().__init__(
            "SMOTEN",
            sampling_ratio=sampling_ratio,
            k_neighbors_default=k_neighbors_default,
            n_jobs=n_jobs,
            random_seed=random_seed,
            **kwargs
        )
