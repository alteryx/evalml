import woodwork as ww
from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class OutliersDataCheck(DataCheck):
    """Checks if there are any outliers in input data by using IQR to determine score anomalies. Columns with score anomalies are considered to contain outliers."""

    def validate(self, X, y=None):
        """Checks if there are any outliers in a dataframe by using IQR to determine column anomalies. Column with anomalies are considered to contain outliers.

        Arguments:
            X (pd.DataFrame, np.ndarray): Features
            y (pd.Series, np.ndarray): Ignored.

        Returns:
            dict: A dictionary with warnings if any columns have outliers.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'x': [1, 2, 3, 4, 5],
            ...     'y': [6, 7, 8, 9, 10],
            ...     'z': [-1, -2, -3, -1201, -4]
            ... })
            >>> outliers_check = OutliersDataCheck()
            >>> assert outliers_check.validate(df) == {"warnings": [{"message": "Column(s) 'z' are likely to have outlier data.",\
                                                                     "data_check_name": "OutliersDataCheck",\
                                                                     "level": "warning",\
                                                                     "code": "HAS_OUTLIERS",\
                                                                     "details": {"columns": ["z"]}}],\
                                                       "errors": [],\
                                                       "actions": []}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)
        X = X.ww.select("numeric")
        X.ww.drop(X.columns[X.isna().all()].tolist(), inplace=True)

        if len(X.columns) == 0:
            return results

        has_outliers = []
        for col in X.columns:
            col_series = ww.init_series(X[col])
            box_plot_dict = col_series.ww.box_plot_dict()
            if len(box_plot_dict['low_values']) or len(box_plot_dict['high_values']):
                has_outliers.append(col)

        warning_msg = "Column(s) {} are likely to have outlier data.".format(
            ", ".join([f"'{col}'" for col in has_outliers])
        )
        results["warnings"].append(
            DataCheckWarning(
                message=warning_msg,
                data_check_name=self.name,
                message_code=DataCheckMessageCode.HAS_OUTLIERS,
                details={"columns": has_outliers},
            ).to_dict()
        )
        return results
