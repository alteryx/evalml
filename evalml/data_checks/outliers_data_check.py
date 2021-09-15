"""Data check that checks if there are any outliers in input data by using IQR to determine score anomalies."""
import numpy as np
from scipy.stats import gamma

from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class OutliersDataCheck(DataCheck):
    """Checks if there are any outliers in input data by using IQR to determine score anomalies.

    Columns with score anomalies are considered to contain outliers.
    """

    def validate(self, X, y=None):
        """Check if there are any outliers in a dataframe by using IQR to determine column anomalies. Column with anomalies are considered to contain outliers.

        Args:
            X (pd.DataFrame, np.ndarray): Input features.
            y (pd.Series, np.ndarray): Ignored. Defaults to None.

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
            >>> assert outliers_check.validate(df) == {
            ...     "warnings": [{"message": "Column(s) 'z' are likely to have outlier data.",
            ...                   "data_check_name": "OutliersDataCheck",\
            ...                   "level": "warning",
            ...                   "code": "HAS_OUTLIERS",
            ...                   "details": {"columns": ["z"]}}],
            ...     "errors": [],
            ...     "actions": []}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)
        X = X.ww.select("numeric")

        if len(X.columns) == 0:
            return results

        has_outliers = []
        for col in X.columns:
            box_plot_dict = X.ww[col].ww.box_plot_dict()
            num_records = len(X[col])
            pct_outliers = (
                len(box_plot_dict["low_values"]) + len(box_plot_dict["high_values"])
            ) / num_records
            if (
                pct_outliers > 0
                and OutliersDataCheck._no_outlier_prob(num_records, pct_outliers) <= 0.9
            ):
                has_outliers.append(col)

        if not len(has_outliers):
            return results

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

    @staticmethod
    def _no_outlier_prob(num_records: int, pct_outliers: float) -> float:
        """Calculate the probability that there are no true outliers in a numeric (integer or float) column.

        It is based on creating 100,000 samples consisting of a given number of records, and then repeating
        this over a grid of sample sizes. Each value in a sample is drawn from a log normal distribution,
        and then the number of potential outliers in the data is determined using the skew adjusted box plot
        approach based on the medcouple statistic.

        It was observed that the distribution of the percentage of outliers could be described by a gamma distribution,
        with the shape and scale parameters changing with the sample size.
        For each sample size, the shape and scale parameters of the gamma distriubtion were estimated using maximum
        likelihood methods. The set of estimate shape and scale parameters for different sample size were then used
        to fit equations that relate these two parameters to the sample size.

        These equations use a transendental logrithmic functional form that provides a seventh order Taylor series
        approximation to the two true functional relationships, and was estimated using least squares regression.

        Original credit goes to Jad Raad and Dan Putler of Alteryx.

        Args:
            num_records (int): The integer number of non-missing values in a column.
            pct_outliers (float): The percentage of potential outliers in a column.

        Returns:
            float: The probability that no outliers are present in the column.
        """
        # Calculate the shape and scale parameters of the approximate
        # gamma distribution given the number of records in the data.
        # For both measures, the values are are from a least squares regression
        # model
        log_n = np.log(num_records)
        log_shape = (
            25.8218734380722
            + -29.2320460088643 * log_n
            + 14.8228030299864 * log_n ** 2
            + -4.08052512660036 * log_n ** 3
            + 0.641429075842177 * log_n ** 4
            + -0.0571252717322226 * log_n ** 5
            + 0.00268694343911156 * log_n ** 6
            + -5.19415149920567e-05 * log_n ** 7
        )
        shape_param = np.exp(log_shape)
        log_scale = (
            -19.8196822259052
            + 8.5359212447622 * log_n
            + -8.80487628113388 * log_n ** 2
            + 2.27711870991327 * log_n ** 3
            + -0.344443407676357 * log_n ** 4
            + 0.029820831994345 * log_n ** 5
            + -0.00136611527293756 * log_n ** 6
            + 2.56727158170901e-05 * log_n ** 7
        )
        scale_param = np.exp(log_scale)

        # calculate and return the probability of no true outliers for a gamma
        # cumulative density function
        prob_val = 1.0 - gamma.cdf(pct_outliers, shape_param, scale=scale_param)
        return prob_val
