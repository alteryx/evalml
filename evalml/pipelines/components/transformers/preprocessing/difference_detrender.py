from skopt.space import Integer
import pandas as pd
import numpy as np
import woodwork as ww

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)


class DifferenceDetrender(Transformer):
    """Removes trends from time series by taking successive differences of the target variable."""
    name = "Difference Detrender"

    hyperparameter_ranges = {}

    def __init__(self, degree=1, random_state=None, random_seed=0, **kwargs):
        """Initialzie the DifferenceDetrender.

        Arguments:
            degree (int): Number of differences to take.
            random_state (None): Deprecated. Use random seed.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        if degree != 1:
            raise ValueError("Degree must be equal to 1!")
        params = {"degree": degree}
        params.update(kwargs)
        self.degree = degree
        self._first_elements = None

        super().__init__(parameters=params,
                         component_obj=None,
                         random_state=random_state,
                         random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the DifferenceDetrender.

        Arguments:
            X (ww.DataTable, pd.DataFrame, optional): Ignored.
            y (ww.DataColumn, pd.Series): Target variable to detrend. Ignored.

        Returns:
            self
        """
        if y is None:
            raise ValueError("y cannot be None for DifferenceDetrender!")
        return self

    def transform(self, X, y=None):
        """Removes fitted trend from target variable.

        Arguments:
            X (ww.DataTable, pd.DataFrame, optional): Ignored.
            y (ww.DataColumn, pd.Series): Target variable to detrend.

        Returns:
            tuple of ww.DataTable, ww.DataColumn: The input features are returned without modification. The target
                variable y is detrended.
        """
        if y is None:
           return X, y
        y_dt = infer_feature_types(y)
        y_t = _convert_woodwork_types_wrapper(y_dt.to_series())
        self._first_elements = y_t.copy(deep=True)
        for degree in range(self.degree):
            # self._first_elements.append(y_t.iloc[degree])
            y_t = y_t - y_t.shift(1)
        y_t = ww.DataColumn(y_t)
        return X, y_t

    def fit_transform(self, X, y=None):
        """Removes fitted trend from target variable.

        Arguments:
            X (ww.DataTable, pd.DataFrame, optional): Ignored.
            y (ww.DataColumn, pd.Series): Target variable to detrend.

        Returns:
            tuple of ww.DataTable, ww.DataColumn: The first element are the input features returned without modification.
                The second element is the target variable y with the fitted trend removed.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y):
        """Adds back fitted trend to target variable.

        Arguments:
            X (ww.DataTable, pd.DataFrame, optional): Ignored.
            y (ww.DataColumn, pd.Series): Target variable.

        Returns:
            tuple of ww.DataTable, ww.DataColumn: The first element are the input features returned without modification.
                The second element is the target variable y with the trend added back.
        """
        if y is None:
            raise ValueError("y cannot be None for DifferenceDetrender!")
        y_dt = infer_feature_types(y)
        y_array = _convert_woodwork_types_wrapper(y_dt.to_series())
        previous_index = max(self._first_elements.index.get_loc(y_array.index[0]) - 1, 0)
        # y_no_nan = y_array.dropna()
        #y_array = y_array.values[self.degree:]
        #for degree in range(self.degree, 0, -1):
        original = np.r_[self._first_elements.iloc[previous_index], y_array.loc[self._first_elements.index[previous_index + 1]:]].cumsum()
        original = pd.Series(original, index=self._first_elements.index[previous_index:])
        original = original.loc[y_array.index[0]:]
        #y_t = ww.DataColumn(pd.Series(y_array, index=y_index))
        y_t = ww.DataColumn(original)
        return X, y_t

