from evalml.pipelines.components.transformers import Transformer
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
