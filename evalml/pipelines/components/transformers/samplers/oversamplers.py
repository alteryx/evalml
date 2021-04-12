from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseOverSampler
)


class SMOTESampler(BaseOverSampler):
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
        parameters = {"sampling_ratio": sampling_ratio,
                      "sampling_ratio_dict": sampling_ratio_dict,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        super().__init__("SMOTE",
                         parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)


class SMOTENCSampler(BaseOverSampler):
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
        parameters = {"categorical_features": categorical_features,
                      "sampling_ratio": sampling_ratio,
                      "sampling_ratio_dict": sampling_ratio_dict,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        super().__init__("SMOTENC",
                         parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)


class SMOTENSampler(BaseOverSampler):
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
        parameters = {"sampling_ratio": sampling_ratio,
                      "sampling_ratio_dict": sampling_ratio_dict,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs}
        parameters.update(kwargs)
        super().__init__("SMOTEN",
                         parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)
