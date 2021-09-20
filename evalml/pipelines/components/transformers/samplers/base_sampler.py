"""Base Sampler component. Used as the base class of all sampler components."""
import copy
from abc import abstractmethod

from evalml.pipelines.components.transformers import Transformer
from evalml.utils.woodwork_utils import infer_feature_types


class BaseSampler(Transformer):
    """Base Sampler component. Used as the base class of all sampler components.

    Args:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    modifies_features = True
    modifies_target = True
    training_only = True

    def fit(self, X, y):
        """Fits the sampler to the data.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target.

        Returns:
            self

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise ValueError("y cannot be None")
        X_ww, y_ww = self._prepare_data(X, y)
        self._initialize_sampler(X_ww, y_ww)
        return self

    @abstractmethod
    def _initialize_sampler(self, X, y):
        """Helper function to initialize the sampler component object.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): The target data.
        """

    def _prepare_data(self, X, y):
        """Transforms the input data to pandas data structure that our sampler can ingest.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target.

        Returns:
            pd.DataFrame, pd.Series: Prepared X and y data as pandas types
        """
        X = infer_feature_types(X)
        if y is None:
            raise ValueError("y cannot be None")
        y = infer_feature_types(y)
        return X, y

    def transform(self, X, y=None):
        """Transforms the input data by sampling the data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target.

        Returns:
            pd.DataFrame, pd.Series: Transformed features and target.
        """
        X_pd, y_pd = self._prepare_data(X, y)
        X_new, y_new = self._component_obj.fit_resample(X_pd, y_pd)
        return infer_feature_types(X_new), infer_feature_types(y_new)

    def _convert_dictionary(self, sampling_dict, y):
        """Converts the provided sampling dictionary from a dictionary of ratios to a dictionary of number of samples.

        Expects the provided dictionary keys to be the target values y, and the associated values to be the min:max ratios.
        Converts and returns a dictionary with the same keys, but changes the values to be the number of samples rather than ratio.

        Args:
            sampling_dict (dict): The input sampling dictionary passed in from user.
            y (pd.Series): The target values.

        Returns:
            dict: A dictionary with target values as keys and the number of samples as values.
        """
        # check that the lengths of the dict and y are equal
        y_unique = y.unique()
        if len(sampling_dict) != len(y_unique):
            raise ValueError(
                "Sampling dictionary contains a different number of targets than are provided in the data."
            )

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
        """If a sampling ratio dictionary is provided, add the updated sampling dictionary to the parameters and return the updated parameter dictionary. Otherwise, simply return the current parameters.

        Args:
            sampling_dict (dict): The input sampling dictionary passed in from user.
            y (pd.Series): The target values.

        Returns:
            dict: The parameters dictionary with the sampling_ratio_dict value replaced as necessary.
        """
        param_copy = copy.copy(self.parameters)
        if self.parameters["sampling_ratio_dict"]:
            new_dic = self._convert_dictionary(
                self.parameters["sampling_ratio_dict"], y
            )
            param_copy["sampling_ratio_dict"] = new_dic
        return param_copy

    def fit_transform(self, X, y):
        """Fit and transform data using the sampler component.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            (pd.DataFrame, pd.Series): Transformed data.
        """
        return self.fit(X, y).transform(X, y)
