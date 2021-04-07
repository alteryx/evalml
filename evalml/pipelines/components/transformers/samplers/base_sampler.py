import copy

from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.utils import make_balancing_dictionary
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)


class BaseSampler(Transformer):
    """Base Sampler component. Used as the base class of all sampler components"""

    def fit(self, X, y):
        """Resample the data using the sampler. Since our sampler doesn't need to be fit, we do nothing here.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

        Returns:
            self
        """
        if y is None:
            raise ValueError("y cannot be none")
        return self

    def _initialize_oversampler(self, X, y, sampler_class):
        """Initializes the oversampler with the given sampler_ratio or sampler_ratio_dict.

        Arguments:
            Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features
            sampler_class (Sampler): The sampler we want to initialize with

        Returns:
            self
        """
        _, _, _, y_pd = self._prepare_data(X, y)
        sampler_params = {k: v for k, v in copy.copy(self.parameters).items() if k not in ['sampling_ratio', 'sampling_ratio_dict']}
        if self.parameters['sampling_ratio_dict'] is not None and len(self.parameters['sampling_ratio_dict']):
            # dictionary provided, which takes priority
            sampler_params['sampling_strategy'] = self.parameters['sampling_ratio_dict']
        else:
            sampling_ratio = self.parameters['sampling_ratio']
            # no dictionary provided. We pass the float if we have a binary situation
            if len(y_pd.value_counts()) == 2:
                sampler_params['sampling_strategy'] = sampling_ratio if sampling_ratio != 1 else 'auto'
            else:
                # otherwise, we make the dictionary
                dic = make_balancing_dictionary(y_pd, sampling_ratio)
                sampler_params['sampling_strategy'] = dic
        sampler = sampler_class(**sampler_params, random_state=self.random_seed)
        self._component_obj = sampler
        return self

    def _prepare_data(self, X, y):
        """Transforms the input data to pandas data structure that our sampler can ingest.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

         Returns:
            ww.DataTable, ww.DataColumn, pd.DataFrame, pd.Series: Prepared X and y data, both woodwork and pandas
        """
        X = infer_feature_types(X)
        if y is None:
            raise ValueError("y cannot be none")
        y = infer_feature_types(y)
        X_pd = _convert_woodwork_types_wrapper(X.to_dataframe())
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        return X, y, X_pd, y_pd

    def transform(self, X, y=None):
        """No transformation needs to be done here.

        Arguments:
            X (ww.DataFrame): Training features. Ignored.
            y (ww.DataColumn): Target features. Ignored.

        Returns:
            ww.DataTable, ww.DataColumn: X and y data that was passed in.
        """
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        return X, y
