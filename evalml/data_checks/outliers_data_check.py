import pandas as pd

from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types
import numpy as np
from scipy.stats import gamma
import sys
from typing import List, Union

import numpy as np

class OutliersDataCheck(DataCheck):
    """Checks if there are any outliers in input data by using IQR to determine score anomalies. Columns with score anomalies are considered to contain outliers."""

    def __init__(self):
        """Checks if there are any outliers in the input data."""

    def no_outlier_prob(self, num_records: int, pct_outliers: float) -> float:
        """
        This functions calculates the probability that there are no true
        outliers in a numeric (integer or float) field. It is based on creating
        samples 100,000 samples consisting of a given number of records, and
        then repeating this over a grid of sample sizes. Each value in a sample
        is drawn from a log normal distribution, and then the number of
        potential outliers in the data is determined using the skew adjusted box
        plot approach based on the medcouple statistic. It was observed that the
        distribution of the percentage of outliers could be described by a gamma
        distribution, with the shape and scale parameters changing with the
        sample size. For each sample size, the shape and scale parameters of the
        gamma distriubtion were estimated using maximum likelihood methods. The
        set of estimate shape and scale parameters for different sample size were
        then used to fit equations that relate these two parameters to the sample
        size. These equations use a transendental logrithmic functional form that
        provides a seventh order Taylor series approximation to the two true
        functional relationships, and was estimated using least squares
        regression.

        :param num_records: the integer number of non-missing values in a field
        :param pct_outliers: the percentage of potential outliers in a field
        :return: the probability that no outliers are present in the field
        """

        # calculate the shape and scale parameters of the approximate
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
            + 18.5359212447622 * log_n
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


    def outlier_score(self, field: pd.Series, convert_field=False) -> dict:
        """
        This function returns a dictionary of high and low values of potential
        numeric outliers using the IQR method.

        :param field: array_like
        :param convert_field: boolean, converts field to int when true
        :return: dictionary of outlier information
        """
        field_nonan = field[~pd.isna(field)]

        if field_nonan.shape[0] == 0:
            return None
        else:
            if convert_field:
                field_nonan = field_nonan.astype(np.int64)

            q1, median, q3 = np.percentile(field_nonan, [25, 50, 75])
            field_iqr = q3 - q1

            low_bound = q1 - (field_iqr * 1.5)
            high_bound = q3 + (field_iqr * 1.5)

            # Filters
            low_filter = field_nonan < low_bound
            high_filter = field_nonan > high_bound

            # Indices
            low_indices = field_nonan[low_filter].index.tolist()
            high_indices = field_nonan[high_filter].index.tolist()

            # Values
            low_values = field.filter(low_indices).tolist()
            high_values = field.filter(high_indices).tolist()

            # calculate outlier probability
            pct_outliers = (len(low_values) + len(high_values)) / len(field_nonan)

            # read in model and retrieve results
            num_records = len(field_nonan)
            score = self.no_outlier_prob(num_records, pct_outliers)

            payload = {
                "score": score,
                "values": {
                    "q1": q1,
                    "median": median,
                    "q3": q1,
                    "low_bound": low_bound,
                    "high_bound": high_bound,
                    "low_values": low_values,
                    "high_values": high_values,
                    "low_indices": low_indices,
                    "high_indices": high_indices,
                },
            }

            return payload



    def validate(self, X, y=None):
        """Checks if there are any outliers in a dataframe by using IQR to determine column anomalies. Column with anomalies are considered to contain outliers.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features
            y (ww.DataColumn, pd.Series, np.ndarray): Ignored.

        Returns:
            dict: A dictionary with warnings if any columns have outliers.

        Example:
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
                                                       "errors": []}
        """
        messages = {
            "warnings": [],
            "errors": []
        }

        X = infer_feature_types(X)
        X = X.select('numeric')
        X = _convert_woodwork_types_wrapper(X.to_dataframe())

        if len(X.columns) == 0:
            return messages

        for col in X.columns:
            self.outlier_score(X[col], False)
        # cols = list(has_outliers.index[has_outliers])
        # warning_msg = "Column(s) {} are likely to have outlier data.".format(", ".join([f"'{col}'" for col in cols]))
        # messages["warnings"].append(DataCheckWarning(message=warning_msg,
                                                    #  data_check_name=self.name,
                                                    #  message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                                    #  details={"columns": cols}).to_dict())
        return messages




    # def validate(self, X, y=None):
    #     """Checks if there are any outliers in a dataframe by using IQR to determine column anomalies. Column with anomalies are considered to contain outliers.

    #     Arguments:
    #         X (ww.DataTable, pd.DataFrame, np.ndarray): Features
    #         y (ww.DataColumn, pd.Series, np.ndarray): Ignored.

    #     Returns:
    #         dict: A dictionary with warnings if any columns have outliers.

    #     Example:
    #         >>> df = pd.DataFrame({
    #         ...     'x': [1, 2, 3, 4, 5],
    #         ...     'y': [6, 7, 8, 9, 10],
    #         ...     'z': [-1, -2, -3, -1201, -4]
    #         ... })
    #         >>> outliers_check = OutliersDataCheck()
    #         >>> assert outliers_check.validate(df) == {"warnings": [{"message": "Column(s) 'z' are likely to have outlier data.",\
    #                                                                  "data_check_name": "OutliersDataCheck",\
    #                                                                  "level": "warning",\
    #                                                                  "code": "HAS_OUTLIERS",\
    #                                                                  "details": {"columns": ["z"]}}],\
    #                                                    "errors": []}
    #     """
    #     messages = {
    #         "warnings": [],
    #         "errors": []
    #     }

    #     X = infer_feature_types(X)
    #     X = X.select('numeric')
    #     X = _convert_woodwork_types_wrapper(X.to_dataframe())

    #     if len(X.columns) == 0:
    #         return messages

    #     def get_IQR(df, k=2.0):
    #         q1 = df.quantile(0.25)
    #         q3 = df.quantile(0.75)
    #         iqr = q3 - q1
    #         lower_bound = pd.Series(q1 - (k * iqr), name='lower_bound')
    #         upper_bound = pd.Series(q3 + (k * iqr), name='upper_bound')
    #         return pd.concat([lower_bound, upper_bound], axis=1)

    #     iqr = get_IQR(X, k=2.0)
    #     has_outliers = ((X < iqr['lower_bound']) | (X > iqr['upper_bound'])).any()
    #     cols = list(has_outliers.index[has_outliers])
    #     warning_msg = "Column(s) {} are likely to have outlier data.".format(", ".join([f"'{col}'" for col in cols]))
    #     messages["warnings"].append(DataCheckWarning(message=warning_msg,
    #                                                  data_check_name=self.name,
    #                                                  message_code=DataCheckMessageCode.HAS_OUTLIERS,
    #                                                  details={"columns": cols}).to_dict())
    #     return messages
