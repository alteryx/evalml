from skopt.space import Integer

from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseSampler
)
from evalml.utils import import_or_raise
from evalml.utils.woodwork_utils import infer_feature_types


class SMOTESampler(BaseSampler):
    """SMOTE Oversampler component. Works on numerical datasets only. This component is only run during training and not during predict."""
    name = "SMOTE Oversampler"
    hyperparameter_ranges = {
        'k_neighbors': Integer(2, 10)
    }

    def __init__(self, sampling_strategy='auto', k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        """Initialize the SMOTE Oversampler component.

        Arguments:
            sampling_strategy (float, dictionary): If float, this is the goal ratio of the minority to majority class.
                A value of 0.25 means we want a 1:4 ratio of the minority to majority class after oversampling.
                If dictionary, the keys correspond to each class, and the values are the number of samples we want to oversample to for each class key.
                Defaults to 'auto', which oversamples all classes to become equal.
                Note that float is only compatible with binary classification problems, and dictionary or 'auto' should be used for multiclass.
            k_neighbors (int): The number of nearest neighbors to used to construct synthetic samples. Defaults to 5.
            n_jobs (int): The number of CPU cores to use. Defaults to -1.
        """
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        parameters = {"sampling_strategy": sampling_strategy,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        sampler = im.SMOTE(**parameters, random_state=random_seed)
        super().__init__(parameters=parameters,
                         component_obj=sampler,
                         random_seed=random_seed)

    def fit_transform(self, X, y):
        """Fit and transform the data using the undersampler. Used during training of the pipeline

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

         Returns:
            ww.DataTable, ww.DataColumn: Undersampled X and y data
        """
        X, y, X_pd, y_pd = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)


class SMOTENCSampler(BaseSampler):
    """SMOTENC Oversampler component. Uses SMOTENC to generate synthetic samples. Works on a mix of nomerical and categorical columns.
       This component is only run during training and not during predict."""
    name = "SMOTENC Oversampler"
    hyperparameter_ranges = {
        'k_neighbors': Integer(2, 10)
    }

    def __init__(self, categorical_features=[], sampling_strategy='auto', k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        """Initialize the SMOTENC Oversampler component.

        Arguments:
            categorical_features (list): A list of indices of the categorical columns, or a list of booleans for each column,
                where True represents a categorical column and False represents a numeric. Defaults empty list.
            sampling_strategy (float, dictionary): If float, this is the goal ratio of the minority to majority class.
                A value of 0.25 means we want a 1:4 ratio of the minority to majority class after oversampling.
                If dictionary, the keys correspond to each class, and the values are the number of samples we want to oversample to for each class key.
                Defaults to 'auto', which oversamples all classes to become equal.
                Note that float is only compatible with binary classification problems, and dictionary or 'auto' should be used for multiclass.
            k_neighbors (int): The number of nearest neighbors to used to construct synthetic samples. Defaults to 5.
            n_jobs (int): The number of CPU cores to use. Defaults to -1.
        """
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        parameters = {"categorical_features": categorical_features,
                      "sampling_strategy": sampling_strategy,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        sampler = im.SMOTENC(**parameters, random_state=random_seed)
        super().__init__(parameters=parameters,
                         component_obj=sampler,
                         random_seed=random_seed)

    def fit_transform(self, X, y):
        """Resample the data
        """
        X, y, X_pd, y_pd = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)


class SMOTENSampler(BaseSampler):
    """SMOTEN Oversampler component. Uses SMOTEN to generate synthetic samples. Works for purely categorical datasets.
       This component is only run during training and not during predict."""
    name = "SMOTEN Oversampler"
    hyperparameter_ranges = {
        'k_neighbors': Integer(2, 10)
    }

    def __init__(self, sampling_strategy='auto', k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        """Initialize the SMOTEN Oversampler component.

        Arguments:
            sampling_strategy (float, dictionary): If float, this is the goal ratio of the minority to majority class.
                A value of 0.25 means we want a 1:4 ratio of the minority to majority class after oversampling.
                If dictionary, the keys correspond to each class, and the values are the number of samples we want to oversample to for each class key.
                Defaults to 'auto', which oversamples all classes to become equal.
                Note that float is only compatible with binary classification problems, and dictionary or 'auto' should be used for multiclass.
            k_neighbors (int): The number of nearest neighbors to used to construct synthetic samples. Defaults to 5.
            n_jobs (int): The number of CPU cores to use. Defaults to -1.
        """
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        parameters = {"sampling_strategy": sampling_strategy,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        sampler = im.SMOTEN(**parameters, random_state=random_seed)
        super().__init__(parameters=parameters,
                         component_obj=sampler,
                         random_seed=random_seed)

    def fit_transform(self, X, y):
        """Resample the data
        """
        X, y, X_pd, y_pd = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)
