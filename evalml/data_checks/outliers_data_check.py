import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning

from evalml.utils import get_random_state
from evalml.utils.gen_utils import numeric_dtypes


class OutliersDataCheck(DataCheck):
    """Checks if there are any outliers in input data by using IQR to determine score anomalies. Columns with score anomalies are considered to contain outliers."""

    def __init__(self, random_state=0):
        """Checks if there are any outliers in the input data.

        Arguments:
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self.random_state = get_random_state(random_state)

    def validate(self, X, y=None):
        """Checks if there are any outliers in a dataframe by using IQR to determine column anomalies. Column with anomalies are considered to contain outliers.

        Arguments:
            X (pd.DataFrame): Features
            y: Ignored.

        Returns:
            A set of columns that may have outlier data.

        Example:
            >>> df = pd.DataFrame({
            ...     'x': [1, 2, 3, 4, 5],
            ...     'y': [6, 7, 8, 9, 10],
            ...     'z': [-1, -2, -3, -1201, -4]
            ... })
            >>> outliers_check = OutliersDataCheck()
            >>> assert outliers_check.validate(df) == [DataCheckWarning("Column 'z' is likely to have outlier data", "OutliersDataCheck")]
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.select_dtypes(include=numeric_dtypes)

        if len(X.columns) == 0:
            return []

        def get_IQR(df, k=2.0):
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (k * iqr)
            upper_bound = q3 + (k * iqr)
            return (lower_bound, upper_bound)

        lower_bound, upper_bound = get_IQR(X)
        indices = set()
        # get the columns that fall out of the bounds, which means they contain outliers
        for idx, bound in enumerate([lower_bound, upper_bound]):
            cols_in_range = (X >= bound.values) if idx == 0 else (X <= bound.values)
            cols_in_range = cols_in_range.all()
            outlier_cols = cols_in_range[~cols_in_range].keys()
            indices.update(outlier_cols.tolist())
        # order the columns by how they appear in the dataframe
        indices = sorted(list(indices), key=lambda x: X.columns.tolist().index(x))
        warning_msg = "Column '{}' is likely to have outlier data"
        s = [DataCheckWarning(warning_msg.format(row_index), self.name) for row_index in indices]
        return s
