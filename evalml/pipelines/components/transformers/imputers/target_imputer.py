import pandas as pd
import woodwork as ww

from . import SimpleImputer

from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class TargetImputer(SimpleImputer):
    """Imputes missing target data according to a specified imputation strategy."""
    name = 'Target Imputer'

    def __init__(self, impute_strategy="most_frequent", fill_value=None, random_seed=0, **kwargs):
        """Initalizes an transformer that imputes missing target data according to the specified imputation strategy."

        Arguments:
            impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
               numerical data, and "most_frequent", "constant" for object data types.
            fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
               Defaults to 0 when imputing numerical data and "missing_value" for strings or object data types.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        super().__init__(impute_strategy=impute_strategy, fill_value=fill_value, random_seed=random_seed, **kwargs)

    def fit(self, X, y):
        """Fits imputer to target data. 'None' values are converted to np.nan before imputation and are
            treated as the same.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]. Ignored.
            y (ww.DataColumn, pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        if y is None:
            raise ValueError("y cannot be None")
        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series()).to_frame()

        # Convert all bool dtypes to category for fitting
        if (y.dtypes == bool).all():
            y = y.astype('category')

        self._component_obj.fit(y)
        return self

    def transform(self, X, y):
        """Transforms input target data by imputing missing values. 'None' and np.nan values are treated as the same.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Features. Ignored.
            y (ww.DataColumn, pd.Series): Target data to impute.

        Returns:
            ww.DataColumn: Transformed y
        """
        y_ww = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y_ww.to_series())
        y_df = y.to_frame()

        # Return early since bool dtype doesn't support nans and sklearn errors if all cols are bool
        if (y_df.dtypes == bool).all():
            return X, _retain_custom_types_and_initalize_woodwork(y_ww, y)

        transformed = self._component_obj.transform(y_df)
        if transformed.shape[1] == 0:
            raise RuntimeError("Transformed data is empty")

        y_t = pd.Series(transformed[:, 0], index=y.index)
        return X, _retain_custom_types_and_initalize_woodwork(y_ww, y_t)

    def fit_transform(self, X, y):
        """Fits on y and transforms y

        Arguments:
            X (ww.DataTable, pd.DataFrame): Features. Ignored.
            y (ww.DataColumn, pd.Series): Target data to impute.

        Returns:
            ww.DataColumn: Transformed y
        """
        return self.fit(X, y).transform(X, y)
