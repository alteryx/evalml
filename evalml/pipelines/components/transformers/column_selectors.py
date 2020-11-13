from abc import abstractmethod

import numpy as np

from evalml.pipelines.components.transformers import Transformer
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class ColumnSelector(Transformer):

    def __init__(self, columns=None, random_state=0, **kwargs):
        """Initalizes an transformer that drops specified columns in input data.

        Arguments:
            columns (list(string)): List of column names, used to determine which columns to drop.
        """
        if columns and not isinstance(columns, list):
            raise ValueError(f"Parameter columns must be a list. Received {type(columns)}.")

        parameters = {"columns": columns}
        parameters.update(kwargs)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _check_input_for_columns(self, X):
        cols = self.parameters.get("columns") or []

        if isinstance(X, np.ndarray):
            column_names = range(X.shape[1])
        else:
            column_names = X.columns

        missing_cols = set(cols) - set(column_names)
        if missing_cols:
            raise ValueError(
                "Columns {} not found in input data".format(', '.join(f"'{col_name}'" for col_name in missing_cols))
            )

    @abstractmethod
    def _modify_columns(self, cols, X, y=None):
        """How the transformer modifies the columns of the input data."""

    def fit(self, X, y=None):
        """'Fits' the transformer by checking if the column names are present in the dataset.

        Arguments:
            X (pd.DataFrame): Data to check.
            y (pd.Series, optional): Targets.

        Returns:
            None.
        """

        self._check_input_for_columns(X)
        return self

    def transform(self, X, y=None):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        self._check_input_for_columns(X)

        cols = self.parameters.get("columns") or []
        return self._modify_columns(cols, X, y)

    def fit_transform(self, X, y=None):
        """Fit transformer to data, then transform data.

        Arguments:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Targets.

        Returns:
            pd.DataFrame: Transformed X.
        """

        # transform method already calls fit under the hood.
        self.fit(X, y)
        return self.transform(X, y)


class DropColumns(ColumnSelector):
    """Drops specified columns in input data."""
    name = "Drop Columns Transformer"
    hyperparameter_ranges = {}
    needs_fitting = False

    def _modify_columns(self, cols, X, y=None):
        return X.drop(columns=cols, axis=1)

    def transform(self, X, y=None):
        """Transforms data X by dropping columns.

        Arguments:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Targets.

        Returns:
            pd.DataFrame: Transformed X.
        """
        return super().transform(X, y)


class SelectColumns(ColumnSelector):
    """Selects specified columns in input data."""
    name = "Select Columns Transformer"
    hyperparameter_ranges = {}
    needs_fitting = False

    def _modify_columns(self, cols, X, y=None):
        return X[cols]

    def transform(self, X, y=None):
        """Transforms data X by selecting columns.

        Arguments:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Targets.

        Returns:
            pd.DataFrame: Transformed X.
        """
        return super().transform(X, y)
