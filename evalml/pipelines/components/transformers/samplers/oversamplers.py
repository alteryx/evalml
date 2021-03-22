from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseOversampler
)
from evalml.utils import import_or_raise
from evalml.utils.woodwork_utils import infer_feature_types

_ERROR_MSG = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"


class SMOTEOversampler(BaseOversampler):
    """SMOTE Oversampler component. Works on numerical datasets only. This component is only run during training and not during predict."""
    name = "SMOTE Oversampler"
    hyperparameter_ranges = {}

    def __init__(self, sampling_ratio=0.25, k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        im = import_or_raise("imblearn.over_sampling", error_msg=_ERROR_MSG)
        self.sampler_class = im.SMOTE
        super().__init__(sampling_ratio=sampling_ratio,
                         k_neighbors=k_neighbors,
                         n_jobs=n_jobs,
                         random_seed=random_seed,
                         **kwargs)


class SMOTENCOversampler(BaseOversampler):
    """SMOTENC Oversampler component. Uses SMOTENC to generate synthetic samples. Works on a mix of nomerical and categorical columns.
       Input data must be Woodwork type, and this component is only run during training and not during predict."""
    name = "SMOTENC Oversampler"
    hyperparameter_ranges = {}

    def __init__(self, sampling_ratio=0.25, k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        im = import_or_raise("imblearn.over_sampling", error_msg=_ERROR_MSG)
        self.sampler_class = im.SMOTENC
        self.categorical_features = None
        super().__init__(sampling_ratio=sampling_ratio,
                         k_neighbors=k_neighbors,
                         n_jobs=n_jobs,
                         random_seed=random_seed,
                         **kwargs)

    def _get_categorical(self, X):
        X = infer_feature_types(X)
        self.categorical_features = [i for i, val in enumerate(X.types['Logical Type'].items()) if str(val[1]) in {'Boolean', 'Categorical'}]
        self._parameters['categorical_features'] = self.categorical_features

    def fit(self, X, y):
        # get categorical features first
        self._get_categorical(X)
        super().fit(X, y)


class SMOTENOversampler(BaseOversampler):
    """SMOTEN Oversampler component. Uses SMOTEN to generate synthetic samples. Works for purely categorical datasets.
       This component is only run during training and not during predict."""
    name = "SMOTEN Oversampler"
    hyperparameter_ranges = {}

    def __init__(self, sampling_ratio=0.25, k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        im = import_or_raise("imblearn.over_sampling", error_msg=_ERROR_MSG)
        self.sampler_class = im.SMOTEN
        super().__init__(sampling_ratio=sampling_ratio,
                         k_neighbors=k_neighbors,
                         n_jobs=n_jobs,
                         random_seed=random_seed,
                         **kwargs)
