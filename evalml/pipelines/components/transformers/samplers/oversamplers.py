from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseSampler,
)
from evalml.pipelines.components.utils import make_balancing_dictionary
from evalml.utils import import_or_raise
from evalml.utils.woodwork_utils import infer_feature_types


class Oversampler(BaseSampler):
    """
    SMOTE Oversampler component. Will automatically select whether to use SMOTE, SMOTEN, or SMOTENC based on inputs to the component.

    Arguments:
        sampler (obj): Sampler object to use.
        sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
            of the minority to majority class after oversampling. We will create the a sampling dictionary using this ratio, with the keys corresponding to the class
            and the values responding to the number of samples. Defaults to 0.25.
        sampling_ratio_dict (dict): A dictionary specifying the desired balanced ratio for each target value. For instance, in a binary case where class 1 is the minority, we could specify:
            `sampling_ratio_dict={0: 0.5, 1: 1}`, which means we would undersample class 0 to have twice the number of samples as class 1 (minority:majority ratio = 0.5), and don't sample class 1.
            Overrides sampling_ratio if provided. Defaults to None.
        k_neighbors_default (int): The number of nearest neighbors used to construct synthetic samples. This is the default value used, but the actual k_neighbors value might be smaller
            if there are less samples. Defaults to 5.
        n_jobs (int): The number of CPU cores to use. Defaults to -1.
        random_seed (int): The seed to use for random sampling. Defaults to 0.
    """
    name = "Oversampler"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        sampling_ratio=0.25,
        sampling_ratio_dict=None,
        k_neighbors_default=5,
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        parameters = {
            "sampling_ratio": sampling_ratio,
            "k_neighbors_default": k_neighbors_default,
            "n_jobs": n_jobs,
            "sampling_ratio_dict": sampling_ratio_dict,
        }
        parameters.update(kwargs)
        self.sampler_options = {"SMOTE": im.SMOTE, "SMOTENC": im.SMOTENC, "SMOTEN": im.SMOTEN}
        self.sampler = None
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y):
        X_ww, y_ww = self._prepare_data(X, y)
        sampler_name = self._get_best_oversampler(X_ww)
        self.sampler = self.sampler_options[sampler_name]

        # get categorical features first, if necessary
        if sampler_name == "SMOTENC":
            self._get_categorical(X)
        super().fit(X, y)
        return self

    def _get_best_oversampler(self, X):
        cat_cols = X.ww.select("Categorical").columns
        if len(cat_cols) == X.shape[1]:
            return "SMOTEN"
        elif not len(cat_cols):
            return "SMOTE"
        else:
            return "SMOTENC"

    def _get_categorical(self, X):
        X = infer_feature_types(X)
        self.categorical_features = [
            i
            for i, val in enumerate(X.ww.types["Logical Type"].items())
            if str(val[1]) in {"Boolean", "Categorical"}
        ]
        self._parameters["categorical_features"] = self.categorical_features

    def _initialize_sampler(self, X, y):
        """Initializes the oversampler with the given sampler_ratio or sampler_ratio_dict. If a sampler_ratio_dict is provided, we will opt to use that.
        Otherwise, we use will create the sampler_ratio_dict dictionary.

        Arguments:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target.
        """
        sampler_class = self.sampler
        _, y_pd = self._prepare_data(X, y)
        sampler_params = {
            k: v
            for k, v in self.parameters.items()
            if k not in ["sampling_ratio", "sampling_ratio_dict", "k_neighbors_default"]
        }
        if self.parameters["sampling_ratio_dict"] is not None:
            # make the dictionary
            dic = self._convert_dictionary(self.parameters["sampling_ratio_dict"], y_pd)
        else:
            # create the sampling dictionary
            sampling_ratio = self.parameters["sampling_ratio"]
            dic = make_balancing_dictionary(y_pd, sampling_ratio)
        sampler_params["sampling_strategy"] = dic

        # check for k_neighbors value
        neighbors = self.parameters["k_neighbors_default"]
        min_counts = y_pd.value_counts().values[-1]
        if min_counts == 1:
            raise ValueError(
                f"Minority class needs more than 1 sample to use SMOTE!, received {min_counts} sample"
            )
        if min_counts <= neighbors:
            neighbors = min_counts - 1

        sampler_params["k_neighbors"] = neighbors
        self._parameters["k_neighbors"] = neighbors
        sampler = sampler_class(**sampler_params, random_state=self.random_seed)
        self._component_obj = sampler


# class SMOTEOversampler(BaseOversampler):
#     """SMOTE Oversampler component. Works on numerical datasets only. This component is only run during training and not during predict.

#     Arguments:
#         sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
#             of the minority to majority class after oversampling. We will create the a sampling dictionary using this ratio, with the keys corresponding to the class
#             and the values responding to the number of samples. Defaults to 0.25.
#         k_neighbors_default (int): The number of nearest neighbors used to construct synthetic samples. This is the default value used, but the actual k_neighbors value might be smaller
#             if there are less samples. Defaults to 5.
#         n_jobs (int): The number of CPU cores to use. Defaults to -1.
#         random_seed (int): The seed to use for random sampling. Defaults to 0.
#     """

#     name = "SMOTE Oversampler"
#     hyperparameter_ranges = {}
#     """{}"""

#     def __init__(
#         self,
#         sampling_ratio=0.25,
#         k_neighbors_default=5,
#         n_jobs=-1,
#         random_seed=0,
#         **kwargs,
#     ):
#         super().__init__(
#             "SMOTE",
#             sampling_ratio=sampling_ratio,
#             k_neighbors_default=k_neighbors_default,
#             n_jobs=n_jobs,
#             random_seed=random_seed,
#             **kwargs,
#         )


# class SMOTENCOversampler(BaseOversampler):
#     """SMOTENC Oversampler component. Uses SMOTENC to generate synthetic samples. Works on a mix of numerical and categorical columns.
#     Input data must be Woodwork type, and this component is only run during training and not during predict.

#     Arguments:
#         sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
#             of the minority to majority class after oversampling. We will create the a sampling dictionary using this ratio, with the keys corresponding to the class
#             and the values responding to the number of samples. Defaults to 0.25.
#         k_neighbors_default (int): The number of nearest neighbors used to construct synthetic samples. This is the default value used, but the actual k_neighbors value might be smaller
#             if there are less samples. Defaults to 5.
#         n_jobs (int): The number of CPU cores to use. Defaults to -1.
#         random_seed (int): The seed to use for random sampling. Defaults to 0.
#     """

#     name = "SMOTENC Oversampler"
#     hyperparameter_ranges = {}
#     """{}"""

#     def __init__(
#         self,
#         sampling_ratio=0.25,
#         k_neighbors_default=5,
#         n_jobs=-1,
#         random_seed=0,
#         **kwargs,
#     ):
#         self.categorical_features = None
#         super().__init__(
#             "SMOTENC",
#             sampling_ratio=sampling_ratio,
#             k_neighbors_default=k_neighbors_default,
#             n_jobs=n_jobs,
#             random_seed=random_seed,
#             **kwargs,
#         )

#     def _get_categorical(self, X):
#         X = infer_feature_types(X)
#         self.categorical_features = [
#             i
#             for i, val in enumerate(X.ww.types["Logical Type"].items())
#             if str(val[1]) in {"Boolean", "Categorical"}
#         ]
#         self._parameters["categorical_features"] = self.categorical_features

#     def fit(self, X, y):
#         # get categorical features first
#         self._get_categorical(X)
#         super().fit(X, y)
#         return self


# class SMOTENOversampler(BaseOversampler):
#     """
#     SMOTEN Oversampler component. Uses SMOTEN to generate synthetic samples. Works for purely categorical datasets.
#     This component is only run during training and not during predict.

#     Arguments:
#         sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
#             of the minority to majority class after oversampling. We will create the a sampling dictionary using this ratio, with the keys corresponding to the class
#             and the values responding to the number of samples. Defaults to 0.25.
#         k_neighbors_default (int): The number of nearest neighbors used to construct synthetic samples. This is the default value used, but the actual k_neighbors value might be smaller
#             if there are less samples. Defaults to 5.
#         n_jobs (int): The number of CPU cores to use. Defaults to -1.
#         random_seed (int): The seed to use for random sampling. Defaults to 0.
#     """

#     name = "SMOTEN Oversampler"
#     hyperparameter_ranges = {}
#     """{}"""

#     def __init__(
#         self,
#         sampling_ratio=0.25,
#         k_neighbors_default=5,
#         n_jobs=-1,
#         random_seed=0,
#         **kwargs,
#     ):
#         super().__init__(
#             "SMOTEN",
#             sampling_ratio=sampling_ratio,
#             k_neighbors_default=k_neighbors_default,
#             n_jobs=n_jobs,
#             random_seed=random_seed,
#             **kwargs,
#         )
