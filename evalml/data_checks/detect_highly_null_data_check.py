import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning


class DetectHighlyNullDataCheck(DataCheck):

    def __init__(self, percent_threshold=0.95):
        """TODO

        Arguments:
            percent_threshold(float): Require that percentage of null values to be considered "highly-null", defaults to 0.95
        """
        if percent_threshold < 0 or percent_threshold > 1:
            raise ValueError("percent_threshold must be a float between 0 and 1, inclusive.")
        self.percent_threshold = percent_threshold

    def validate(self, X, y=None):
        """ Checks if there are any highly-null columns in a pd.Dataframe.

        Arguments:
            X (pd.DataFrame) : features
            y : Ignored.

        Returns:
        Example:
            >>> df = pd.DataFrame({
            ...    'lots_of_null': [None, None, None, None, 5],
            ...    'no_null': [1, 2, 3, 4, 5]
            ... })
            >>> null_check = DetectHighlyNullDataCheck(percent_threshold=0.8)
            >>> null_check.validate(df)
        """
        messages = []
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        percent_null = (X.isnull().mean()).to_dict()
        highly_null_cols = {key: value for key, value in percent_null.items() if value >= self.percent_threshold}
        if len(highly_null_cols) > 0:
            col_names_str = ', '.join([f"'{name}'" for name in list(highly_null_cols.keys())])
            warning_msg = "Columns {} are more than {}% null".format(col_names_str, self.percent_threshold * 100.)
            warning = DataCheckWarning(warning_msg, self.name)
            messages.append(warning)
        return messages
