from abc import abstractmethod

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class ColumnSelector(Transformer):

    def __init__(self, columns=None, random_seed=0, **kwargs):
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
                         random_seed=random_seed)

    def _check_input_for_columns(self, X):
        cols = self.parameters.get("columns") or []

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
        """Fits the transformer by checking if column names are present in the dataset.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to check.
            y (ww.DataColumn, pd.Series, optional): Targets.

        Returns:
            self
        """
        X = infer_feature_types(X)
        self._check_input_for_columns(X)
        return self

    def transform(self, X, y=None):
        X = infer_feature_types(X)
        self._check_input_for_columns(X)
        cols = self.parameters.get("columns") or []
        modified_cols = self._modify_columns(cols, X, y)
        return infer_feature_types(modified_cols)


class DropColumns(ColumnSelector):
    """Drops specified columns in input data."""
    name = "Drop Columns Transformer"
    hyperparameter_ranges = {}
    needs_fitting = False

    def _modify_columns(self, cols, X, y=None):
        return X.drop(columns=cols)

    def transform(self, X, y=None):
        """Transforms data X by dropping columns.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to transform.
            y (ww.DataColumn, pd.Series, optional): Targets.

        Returns:
            ww.DataTable: Transformed X.
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
            X (ww.DataTable, pd.DataFrame): Data to transform.
            y (ww.DataColumn, pd.Series, optional): Targets.

        Returns:
            ww.DataTable: Transformed X.
        """
        return super().transform(X, y)
