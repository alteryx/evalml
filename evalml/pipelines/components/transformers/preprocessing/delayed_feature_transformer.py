import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from woodwork import logical_types

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class DelayedFeatureTransformer(Transformer):
    """Transformer that delayes input features and target variable for time series problems."""
    name = "Delayed Feature Transformer"
    hyperparameter_ranges = {}
    needs_fitting = False

    def __init__(self, date_index=None, max_delay=2, delay_features=True, delay_target=True, gap=1, random_seed=0, **kwargs):
        """Creates a DelayedFeatureTransformer.

        Arguments:
            date_index (str): Name of the column containing the datetime information used to order the data. Ignored.
            max_delay (int): Maximum number of time units to delay each feature.
            delay_features (bool): Whether to delay the input features.
            delay_target (bool): Whether to delay the target.
            gap (int): The number of time units between when the features are collected and
                when the target is collected. For example, if you are predicting the next time step's target, gap=1.
                This is only needed because when gap=0, we need to be sure to start the lagging of the target variable
                at 1.
            random_seed (int): Seed for the random number generator. This transformer performs the same regardless of the random seed provided.
        """
        self.date_index = date_index
        self.max_delay = max_delay
        self.delay_features = delay_features
        self.delay_target = delay_target

        # If 0, start at 1
        self.start_delay_for_target = int(gap == 0)

        parameters = {"date_index": date_index, "max_delay": max_delay, "delay_target": delay_target, "delay_features": delay_features,
                      "gap": gap}
        parameters.update(kwargs)
        super().__init__(parameters=parameters, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the DelayFeatureTransformer.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        return self

    @staticmethod
    def _encode_y_while_preserving_index(y):
        original_y = _convert_woodwork_types_wrapper(y.to_series())
        y_encoded = LabelEncoder().fit_transform(original_y)
        y = pd.Series(y_encoded, index=original_y.index)
        return y

    @staticmethod
    def _get_categorical_columns(X):
        return [name for name, column in X.columns.items() if column.logical_type == logical_types.Categorical]

    @staticmethod
    def _encode_X_while_preserving_index(X_categorical):
        return pd.DataFrame(OrdinalEncoder().fit_transform(X_categorical),
                            columns=X_categorical.columns, index=X_categorical.index)

    def transform(self, X, y=None):
        """Computes the delayed features for all features in X and y.

        For each feature in X, it will add a column to the output dataframe for each
        delay in the (inclusive) range [1, max_delay]. The values of each delayed feature are simply the original
        feature shifted forward in time by the delay amount. For example, a delay of 3 units means that the feature
        value at row n will be taken from the n-3rd row of that feature

        If y is not None, it will also compute the delayed values for the target variable.

        Arguments:
            X (ww.DataTable, pd.DataFrame or None): Data to transform. None is expected when only the target variable is being used.
            y (ww.DataColumn, pd.Series, or None): Target.

        Returns:
            ww.DataTable: Transformed X.
        """
        if X is None:
            X = pd.DataFrame()
        # Normalize the data into pandas objects
        X_ww = infer_feature_types(X)
        categorical_columns = self._get_categorical_columns(X_ww)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        if self.delay_features and len(X) > 0:
            X_categorical = self._encode_X_while_preserving_index(X[categorical_columns])
            for col_name in X:
                col = X[col_name]
                if col_name in categorical_columns:
                    col = X_categorical[col_name]
                X = X.assign(**{f"{col_name}_delay_{t}": col.shift(t) for t in range(1, self.max_delay + 1)})
        # Handle cases where the target was passed in
        if self.delay_target and y is not None:
            y = infer_feature_types(y)
            if y.logical_type == logical_types.Categorical:
                y = self._encode_y_while_preserving_index(y)
            else:
                y = _convert_woodwork_types_wrapper(y.to_series())
            X = X.assign(**{f"target_delay_{t}": y.shift(t)
                            for t in range(self.start_delay_for_target, self.max_delay + 1)})

        return _retain_custom_types_and_initalize_woodwork(X_ww, X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
