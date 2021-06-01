import copy

from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.utils import make_balancing_dictionary
from evalml.utils import import_or_raise
from evalml.utils.woodwork_utils import infer_feature_types


class BaseSampler(Transformer):
    """Base Sampler component. Used as the base class of all sampler components"""

    def fit(self, X, y):
        """Resample the data using the sampler. Since our sampler doesn't need to be fit, we do nothing here.

        Arguments:
            X (pd.DataFrame): Training features
            y (pd.Series): Target features

        Returns:
            self
        """
        if y is None:
            raise ValueError("y cannot be none")
        return self

    def _prepare_data(self, X, y):
        """Transforms the input data to pandas data structure that our sampler can ingest.

        Arguments:
            X (pd.DataFrame): Training features
            y (pd.Series): Target features

         Returns:
            pd.DataFrame, pd.Series: Prepared X and y data, both woodwork and pandas
        """
        X = infer_feature_types(X)
        if y is None:
            raise ValueError("y cannot be none")
        y = infer_feature_types(y)
        return X, y

    def transform(self, X, y=None):
        """No transformation needs to be done here.

        Arguments:
            X (pd.DataFrame): Training features. Ignored.
            y (pd.Series): Target features. Ignored.

        Returns:
            pd.DataFrame, pd.Series: X and y data that was passed in.
        """
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        return X, None

    def _convert_dictionary(self, sampling_dict, y):
        """Converts the provided sampling dictionary from a dictionary of ratios to a dictionary of number of samples.
        Expects the provided dictionary keys to be the target values y, and the associated values to be the min:max ratios.
        Converts and returns a dictionary with the same keys, but changes the values to be the number of samples rather than ratio.

        Arguments:
            sampling_dict (dict): The input sampling dictionary passed in from user
            y (pd.Series): The target values

        Returns:
            dict: A dictionary with target values as keys and the number of samples as values
        """
        # check that the lengths of the dict and y are equal
        y_unique = y.unique()
        if len(sampling_dict) != len(y_unique):
            raise ValueError("Sampling dictionary contains a different number of targets than are provided in the data.")

        if len(set(sampling_dict.keys()).intersection(set(y_unique))) != len(y_unique):
            raise ValueError("Dictionary keys are different from target values!")

        new_dic = {}
        y_counts = y.value_counts()
        for k, v in sampling_dict.items():
            # turn the ratios into sampler values
            if self.__class__.__name__ == "Undersampler":
                # for undersampling, we make sure we never sample more than the
                # total samples for that class
                new_dic[k] = int(min(y_counts.values[-1] / v, y_counts[k]))
            else:
                # for oversampling, we need to make sure we never sample less than
                # the total samples for that class
                new_dic[k] = int(max(y_counts.values[0] * v, y_counts[k]))
        return new_dic

    def _dictionary_to_params(self, sampling_dict, y):
        """If a sampling ratio dictionary is provided, add the updated sampling dictionary to the
        parameters and return the updated parameter dictionary. Otherwise, simply return the current parameters.

        Arguments:
            sampling_dict (dict): The input sampling dictionary passed in from user
            y (pd.Series): The target values

        Returns:
            dict: The parameters dictionary with the sampling_ratio_dict value replaced as necessary
        """
        param_copy = copy.copy(self.parameters)
        if self.parameters['sampling_ratio_dict']:
            new_dic = self._convert_dictionary(self.parameters['sampling_ratio_dict'], y)
            param_copy['sampling_ratio_dict'] = new_dic
        return param_copy


class BaseOverSampler(BaseSampler):
    """Base Oversampler component. Used as the base class of all imbalance-learn oversampler components"""

    def __init__(self, sampler, sampling_ratio=0.25, sampling_ratio_dict=None, k_neighbors=5, n_jobs=-1, random_seed=0, **kwargs):
        """Initializes the oversampler component.

        Arguments:
            sampling_ratio (float): This is the goal ratio of the minority to majority class, with range (0, 1]. A value of 0.25 means we want a 1:4 ratio
                of the minority to majority class after oversampling. We will create the a sampling dictionary using this ratio, with the keys corresponding to the class
                and the values responding to the number of samples. Defaults to 0.25.
            k_neighbors (int): The number of nearest neighbors to used to construct synthetic samples. Defaults to 5.
            n_jobs (int): The number of CPU cores to use. Defaults to -1.
        """
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        parameters = {"sampling_ratio": sampling_ratio,
                      "k_neighbors": k_neighbors,
                      "n_jobs": n_jobs,
                      "sampling_ratio_dict": sampling_ratio_dict}
        parameters.update(kwargs)
        self.sampler = {"SMOTE": im.SMOTE,
                        "SMOTENC": im.SMOTENC,
                        "SMOTEN": im.SMOTEN}[sampler]
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Fits the Oversampler to the data.

        Arguments:
            X (pd.DataFrame): Training features
            y (pd.Series): Target features

        Returns:
            self
        """
        super().fit(X, y)
        self._initialize_oversampler(X, y, self.sampler)

    def _initialize_oversampler(self, X, y, sampler_class):
        """Initializes the oversampler with the given sampler_ratio or sampler_ratio_dict. If a sampler_ratio_dict is provided, we will opt to use that.
        Otherwise, we use will create the sampler_ratio_dict dictionary.

        Arguments:
            X (pd.DataFrame): Training features
            y (pd.Series): Target features
            sampler_class (imblearn.BaseSampler): The sampler we want to initialize
        """
        _, y_ww = self._prepare_data(X, y)
        sampler_params = {k: v for k, v in copy.copy(self.parameters).items() if k not in ['sampling_ratio', 'sampling_ratio_dict']}
        if self.parameters['sampling_ratio_dict'] is not None:
            # make the dictionary
            dic = self._convert_dictionary(self.parameters['sampling_ratio_dict'], y_ww)
        else:
            # create the sampling dictionary
            sampling_ratio = self.parameters['sampling_ratio']
            dic = make_balancing_dictionary(y_ww, sampling_ratio)
        sampler_params['sampling_strategy'] = dic
        sampler = sampler_class(**sampler_params, random_state=self.random_seed)
        self._component_obj = sampler

    def fit_transform(self, X, y):
        """Fit and transform the data using the data sampler. Used during training of the pipeline

        Arguments:
            X (pd.DataFrame): Training features
            y (pd.Series): Target features

         Returns:
            pd.DataFrame, pd.Series: Sampled X and y data
        """
        self.fit(X, y)
        X_ww, y_ww = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_ww, y_ww)
        return infer_feature_types(X_new), infer_feature_types(y_new)
