"""Component that imputes missing data according to a specified imputation strategy."""
import numpy as np
import pandas as pd
import woodwork
from sklearn.impute import KNNImputer as Sk_KNNImputer

from evalml.pipelines.components.transformers import Transformer
from evalml.pipelines.components.utils import drop_natural_language_columns
from evalml.utils import infer_feature_types


class KNNImputer(Transformer):
    """Imputes missing data using KNN according to a specified number of neighbors.  Natural language columns are ignored.

    Args:
        number_neighbors (int): Number of nearest neighbors for KNN to search for. Defaults to 3.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    """

    name = "KNN Imputer"

    def __init__(self, number_neighbors=3, random_seed=0, **kwargs):
        parameters = {"number_neighbors": number_neighbors}
        parameters.update(kwargs)

        imputer = Sk_KNNImputer(
            n_neighbors=number_neighbors,
            missing_values=np.nan,
            **kwargs,
        )
        self._all_null_cols = None
        super().__init__(
            parameters=parameters,
            component_obj=imputer,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits imputer to data. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame or np.ndarray): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training data of length [n_samples]

        Returns:
            self

        Raises:
            ValueError: if the KNNImputer receives a dataframe with both Boolean and Categorical data.

        """
        X = infer_feature_types(X)
        X, _ = drop_natural_language_columns(X)

        nan_ratio = X.isna().sum() / X.shape[0]
        self._all_null_cols = nan_ratio[nan_ratio == 1].index.tolist()

        # If the Dataframe only had natural language columns, do nothing.
        if X.shape[1] == 0:
            return self

        self._component_obj.fit(X, y)
        return self

    def transform(self, X, y=None):
        """Transforms input by imputing missing values. 'None' and np.nan values are treated as the same.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X = infer_feature_types(X)

        not_all_null_cols = [col for col in X.columns if col not in self._all_null_cols]
        original_index = X.index

        # Drop natural language columns and transform the other columns
        X_t, natural_language_cols = drop_natural_language_columns(X)
        if X_t.shape[1] == 0:
            return X
        not_all_null_or_natural_language_cols = [
            col for col in not_all_null_cols if col not in natural_language_cols
        ]

        X_t = self._component_obj.transform(X_t)
        X_t = pd.DataFrame(X_t, columns=not_all_null_or_natural_language_cols)

        X_schema = X.ww.schema

        X_bool_nullable_cols = X_schema._filter_cols(include=["BooleanNullable"])
        X_int_nullable_cols = X_schema._filter_cols(include=["IntegerNullable"])
        new_ltypes_for_nullable_cols = {col: "Boolean" for col in X_bool_nullable_cols}
        new_ltypes_for_nullable_cols.update(
            {col: "Double" for col in X_int_nullable_cols},
        )

        # Add back in natural language columns, unchanged
        if len(natural_language_cols) > 0:
            X_t = woodwork.concat_columns([X_t, X[natural_language_cols]])

        X_t.ww.init(
            schema=X_schema,
            logical_types=new_ltypes_for_nullable_cols,
        )

        if not_all_null_or_natural_language_cols:
            X_t.index = original_index

        return X_t

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X.

        Args:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series, optional): Target data.

        Returns:
            pd.DataFrame: Transformed X
        """
        return self.fit(X, y).transform(X, y)
