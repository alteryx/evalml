import pandas as pd
from sklearn.ensemble import IsolationForest

from .data_check import DataCheck
from .data_check_message import DataCheckWarning


class DetectOutliersDataCheck(DataCheck):

    def __init__(self, random_state=0):
        """Checks if there are any outliers in a DataFrame by using first Isolation Forest to obtain the anomaly score
        of each index and then using IQR to determine score anomalies. Indices with score anomalies are considered outliers.

        Arguments:
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self.random_state = random_state

    def validate(self, X, y=None):
        """Checks if there are any outliers in a dataframe by using first Isolation Forest to obtain the anomaly score
        of each index and then using IQR to determine score anomalies. Indices with score anomalies are considered outliers.

        Arguments:
            X (pd.DataFrame): features
            y: Ignored.

        Returns:
            A set of indices that may have outlier data.

        Example:
            >>> df = pd.DataFrame({
            ...     'x': [1, 2, 3, 40, 5],
            ...     'y': [6, 7, 8, 990, 10],
            ...     'z': [-1, -2, -3, -1201, -4]
            ... })
            >>> outliers_check = DetectOutliersDataCheck()
            >>> assert outliers_check.validate(df) == [DataCheckWarning("Row '3' is likely to have outlier data", "DetectOutliersDataCheck")]
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # only select numeric
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        X = X.select_dtypes(include=numerics)

        if len(X.columns) == 0:
            return {}

        def get_IQR(df, k=2.0):
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (k * iqr)
            upper_bound = q3 + (k * iqr)
            return (lower_bound, upper_bound)

        clf = IsolationForest(random_state=self.random_state, behaviour="new", contamination=0.1)
        clf.fit(X)
        scores = pd.Series(clf.decision_function(X))
        lower_bound, upper_bound = get_IQR(scores, k=2)
        outliers = (scores < lower_bound) | (scores > upper_bound)
        outliers_indices = outliers[outliers].index.values.tolist()
        warning_msg = "Row '{}' is likely to have outlier data"
        return [DataCheckWarning(warning_msg.format(row_index), self.name) for row_index in outliers_indices]
