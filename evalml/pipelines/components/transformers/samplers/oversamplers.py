import copy
from skopt.space import Integer

from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseSampler
)
from evalml.pipelines.components.utils import make_balancing_dictionary
from evalml.utils import import_or_raise
from evalml.utils.woodwork_utils import infer_feature_types


class SMOTESampler(BaseSampler):
    """SMOTE Oversampler component. Works on numerical datasets only. This component is only run during training and not during predict."""
    name = "SMOTE Oversampler"

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
        sampler_params = {k: v for k, v in copy.copy(self.parameters).items() if k not in ['sampling_ratio', 'sampling_ratio_dict']}
        if self.parameters['sampling_ratio_dict'] is not None and len(self.parameters['sampling_ratio_dict']):
            # dictionary provided, which takes priority
            sampler_params['sampling_strategy'] = self.parameters['sampling_ratio_dict']
        else:
            sampling_ratio = self.parameters['sampling_ratio']
            # no dictionary provided. We pass the float if we have a binary situation
            if len(y.value_counts()) == 2:
                sampler_params['sampling_strategy'] = sampling_ratio if sampling_ratio != 1 else 'auto'
            else:
                # otherwise, we make the dictionary
                dic = make_balancing_dictionary(y, sampling_ratio)
                sampler_params['sampling_strategy'] = dic
        sampler = self.im.SMOTE(**sampler_params, random_state=self.random_seed)
        self._component_obj = sampler

    def fit_transform(self, X, y):
        """Fit and transform the data using the undersampler. Used during training of the pipeline

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

         Returns:
            ww.DataTable, ww.DataColumn: Undersampled X and y data
        """
        _, _, X_pd, y_pd = self._prepare_data(X, y)
        self.fit(X_pd, y_pd)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)


class SMOTENCSampler(BaseSampler):
    """SMOTENC Oversampler component. Uses SMOTENC to generate synthetic samples. Works on a mix of nomerical and categorical columns.
       This component is only run during training and not during predict."""
    name = "SMOTENC Oversampler"

    def __init__(self, categorical_features=[], sampling_ratio=0.25, sampling_ratio_dict=None, k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        """Initialize the SMOTENC Oversampler component.

        Arguments:
            categorical_features (list): A list of indices of the categorical columns, or a list of booleans for each column,
                where True represents a categorical column and False represents a numeric. Defaults empty list.
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
        sampler_params = {k: v for k, v in copy.copy(self.parameters).items() if k not in ['sampling_ratio', 'sampling_ratio_dict']}
        if self.parameters['sampling_ratio_dict'] is not None and len(self.parameters['sampling_ratio_dict']):
            # dictionary provided, which takes priority
            sampler_params['sampling_strategy'] = self.parameters['sampling_ratio_dict']
        else:
            sampling_ratio = self.parameters['sampling_ratio']
            # no dictionary provided. We pass the float if we have a binary situation
            if len(y.value_counts()) == 2:
                sampler_params['sampling_strategy'] = sampling_ratio if sampling_ratio != 1 else 'auto'
            else:
                # otherwise, we make the dictionary
                dic = make_balancing_dictionary(y, sampling_ratio)
                sampler_params['sampling_strategy'] = dic
        sampler = self.im.SMOTENC(**sampler_params, random_state=self.random_seed)
        self._component_obj = sampler

    def fit_transform(self, X, y):
        """Resample the data
        """
        _, _, X_pd, y_pd = self._prepare_data(X, y)
        self.fit(X_pd, y_pd)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)


class SMOTENSampler(BaseSampler):
    """SMOTEN Oversampler component. Uses SMOTEN to generate synthetic samples. Works for purely categorical datasets.
       This component is only run during training and not during predict."""
    name = "SMOTEN Oversampler"

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
        sampler_params = {k: v for k, v in copy.copy(self.parameters).items() if k not in ['sampling_ratio', 'sampling_ratio_dict']}
        if self.parameters['sampling_ratio_dict'] is not None and len(self.parameters['sampling_ratio_dict']):
            # dictionary provided, which takes priority
            sampler_params['sampling_strategy'] = self.parameters['sampling_ratio_dict']
        else:
            sampling_ratio = self.parameters['sampling_ratio']
            # no dictionary provided. We pass the float if we have a binary situation
            if len(y.value_counts()) == 2:
                sampler_params['sampling_strategy'] = sampling_ratio if sampling_ratio != 1 else 'auto'
            else:
                # otherwise, we make the dictionary
                dic = make_balancing_dictionary(y, sampling_ratio)
                sampler_params['sampling_strategy'] = dic
        sampler = self.im.SMOTEN(**sampler_params, random_state=self.random_seed)
        self._component_obj = sampler

    def fit_transform(self, X, y):
        """Resample the data
        """
        _, _, X_pd, y_pd = self._prepare_data(X, y)
        self.fit(X_pd, y_pd)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)
