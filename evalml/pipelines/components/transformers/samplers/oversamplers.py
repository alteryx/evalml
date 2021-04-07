from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseSampler
)
from evalml.utils import import_or_raise
from evalml.utils.woodwork_utils import infer_feature_types


class SMOTESampler(BaseSampler):
    """SMOTE Oversampler component. Works on numerical datasets only. This component is only run during training and not during predict."""
    name = "SMOTE Oversampler"
    hyperparameter_ranges = {}

    def __init__(self, sampling_ratio=0.25, sampling_ratio_dict=None, k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        """Initialize the SMOTE Oversampler component.

        Arguments:
            sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
                of the minority to majority class after oversampling. If the targets are multiclass, will create a dictionary using this ratio. Defaults to 0.25.
            sampling_ratio_dict (dict): Dictionary which has keys corresponding to each class, and the values are the number of samples we want to oversample to for each class key.
                If this value is provided, it will be used. Otherwise, we opt to use sampling_ratio. Defaults to None.
            k_neighbors (int): The number of nearest neighbors to used to construct synthetic samples. Defaults to 5.
            n_jobs (int): The number of CPU cores to use. Defaults to -1.
        """
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        self.im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        parameters = {"sampling_ratio": sampling_ratio,
                      "sampling_ratio_dict": sampling_ratio_dict,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Fits the sampler to the data.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

        Returns:
            self
        """
        super().fit(X, y)
        super()._initialize_oversampler(X, y, self.im.SMOTE)

    def fit_transform(self, X, y):
        """Fit and transform the data using the undersampler. Used during training of the pipeline

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

         Returns:
            ww.DataTable, ww.DataColumn: Undersampled X and y data
        """
        self.fit(X, y)
        _, _, X_pd, y_pd = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)


class SMOTENCSampler(BaseSampler):
    """SMOTENC Oversampler component. Uses SMOTENC to generate synthetic samples. Works on a mix of nomerical and categorical columns.
       This component is only run during training and not during predict."""
    name = "SMOTENC Oversampler"
    hyperparameter_ranges = {}

    def __init__(self, categorical_features=[], sampling_ratio=0.25, sampling_ratio_dict=None, k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        """Initialize the SMOTENC Oversampler component.

        Arguments:
            categorical_features (list): A list of indices of the categorical columns, or a list of booleans for each column,
                where True represents a categorical column and False represents a numeric. There must exist a mix of both categorical and numeric columns.
                Defaults to an empty list.
            sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
                of the minority to majority class after oversampling. If the targets are multiclass, will create a dictionary using this ratio. Defaults to 0.25.
            sampling_ratio_dict (dict): Dictionary which has keys corresponding to each class, and the values are the number of samples we want to oversample to for each class key.
                If this value is provided, it will be used. Otherwise, we opt to use sampling_ratio. Defaults to None.
            k_neighbors (int): The number of nearest neighbors to used to construct synthetic samples. Defaults to 5.
            n_jobs (int): The number of CPU cores to use. Defaults to -1.
        """
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        self.im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        parameters = {"categorical_features": categorical_features,
                      "sampling_ratio": sampling_ratio,
                      "sampling_ratio_dict": sampling_ratio_dict,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Fits the sampler to the data.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

        Returns:
            self
        """
        super().fit(X, y)
        # Ensure that the data has a mix of both numeric and categorical features
        cat_feat = self.parameters['categorical_features']
        all_unique_indices = len(set(cat_feat)) == len(cat_feat)
        same_length = len(cat_feat) == X.shape[1]
        if len(cat_feat) == 0 or (same_length and (all(cat_feat) or all_unique_indices)) or not any(cat_feat):
            raise ValueError("The length of categorical_features must be longer than 0, but the dataset cannot all be categorical features!")
        super()._initialize_oversampler(X, y, self.im.SMOTENC)

    def fit_transform(self, X, y):
        """Resample the data
        """
        self.fit(X, y)
        _, _, X_pd, y_pd = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)


class SMOTENSampler(BaseSampler):
    """SMOTEN Oversampler component. Uses SMOTEN to generate synthetic samples. Works for purely categorical datasets.
       This component is only run during training and not during predict."""
    name = "SMOTEN Oversampler"
    hyperparameter_ranges = {}

    def __init__(self, sampling_ratio=0.25, sampling_ratio_dict=None, k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        """Initialize the SMOTEN Oversampler component.

        Arguments:
            sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
                of the minority to majority class after oversampling. If the targets are multiclass, will create a dictionary using this ratio. Defaults to 0.25.
            sampling_ratio_dict (dict): Dictionary which has keys corresponding to each class, and the values are the number of samples we want to oversample to for each class key.
                If this value is provided, it will be used. Otherwise, we opt to use sampling_ratio. Defaults to None.
            k_neighbors (int): The number of nearest neighbors to used to construct synthetic samples. Defaults to 5.
            n_jobs (int): The number of CPU cores to use. Defaults to -1.
        """
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        self.im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        parameters = {"sampling_ratio": sampling_ratio,
                      "sampling_ratio_dict": sampling_ratio_dict,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Fits the sampler to the data.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

        Returns:
            self
        """
        super().fit(X, y)
        super()._initialize_oversampler(X, y, self.im.SMOTEN)

    def fit_transform(self, X, y):
        """Resample the data
        """
        self.fit(X, y)
        _, _, X_pd, y_pd = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)
