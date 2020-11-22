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
            dict: A set of columns that may have outlier data.

        Example:
            >>> df = pd.DataFrame({
            ...     'x': [1, 2, 3, 4, 5],
            ...     'y': [6, 7, 8, 9, 10],
            ...     'z': [-1, -2, -3, -1201, -4]
            ... })
            >>> outliers_check = OutliersDataCheck()
            >>> assert outliers_check.validate(df) == {"warnings": [{"message": "Column 'z' is likely to have outlier data", "data_check_name": "OutliersDataCheck", "level": "warning"}],\
                                                       "errors": []}
        """
        messages = {
            "warnings": [],
            "errors": []
        }
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.select_dtypes(include=numeric_dtypes)

        if len(X.columns) == 0:
            return messages

        def get_IQR(df, k=2.0):
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            lower_bound = pd.Series(q1 - (k * iqr), name='lower_bound')
            upper_bound = pd.Series(q3 + (k * iqr), name='upper_bound')
            return pd.concat([lower_bound, upper_bound], axis=1)

        iqr = get_IQR(X, k=2.0)
        has_outliers = ((X < iqr['lower_bound']) | (X > iqr['upper_bound'])).any()
        warning_msg = "Column '{}' is likely to have outlier data"
        cols = has_outliers.index[has_outliers]
        messages["warnings"].extend([DataCheckWarning(warning_msg.format(col), self.name).to_dict() for col in cols])
        return messages
