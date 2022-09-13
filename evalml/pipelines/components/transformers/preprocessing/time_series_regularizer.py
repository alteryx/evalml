"""Transformer that regularizes a dataset with an uninferrable offset frequency for time series problems."""
import pandas as pd
import woodwork as ww
from woodwork.logical_types import Datetime
from woodwork.statistics_utils import infer_frequency

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


class TimeSeriesRegularizer(Transformer):
    """Transformer that regularizes an inconsistently spaced datetime column.

    If X is passed in to fit/transform, the column `time_index` will be checked for an inferrable offset frequency. If
    the `time_index` column is perfectly inferrable then this Transformer will do nothing and return the original X and y.

    If X does not have a perfectly inferrable frequency but one can be estimated, then X and y will be reformatted based
    on the estimated frequency for `time_index`. In the original X and y passed:
    - Missing datetime values will be added and will have their corresponding columns in X and y set to None.
    - Duplicate datetime values will be dropped.
    - Extra datetime values will be dropped.
    - If it can be determined that a duplicate or extra value is misaligned, then it will be repositioned to take the
    place of a missing value.

    This Transformer should be used before the `TimeSeriesImputer` in order to impute the missing values that were
    added to X and y (if passed).

    Args:
        time_index (string): Name of the column containing the datetime information used to order the data, required. Defaults to None.
        frequency_payload (tuple): Payload returned from Woodwork's infer_frequency function where debug is True. Defaults to None.
        window_length (int): The size of the rolling window over which inference is conducted to determine the prevalence of uninferrable frequencies.
        Lower values make this component more sensitive to recognizing numerous faulty datetime values. Defaults to 5.
        threshold (float): The minimum percentage of windows that need to have been able to infer a frequency. Lower values make this component more
        sensitive to recognizing numerous faulty datetime values. Defaults to 0.8.
        random_seed (int): Seed for the random number generator. This transformer performs the same regardless of the random seed provided.
        Defaults to 0.

    Raises:
        ValueError: if the frequency_payload parameter has not been passed a tuple
    """

    name = "Time Series Regularizer"
    hyperparameter_ranges = {}
    """{}"""

    modifies_target = True
    training_only = True

    def __init__(
        self,
        time_index=None,
        frequency_payload=None,
        window_length=4,
        threshold=0.4,
        random_seed=0,
        **kwargs,
    ):
        self.time_index = time_index
        self.frequency_payload = frequency_payload
        self.window_length = window_length
        self.threshold = threshold
        self.error_dict = {}
        self.inferred_freq = None
        self.debug_payload = None

        if self.frequency_payload and not isinstance(self.frequency_payload, tuple):
            raise ValueError(
                "The frequency_payload parameter must be a tuple returned from Woodwork's infer_frequency function where debug is True.",
            )

        parameters = {
            "time_index": time_index,
            "window_length": window_length,
            "threshold": threshold,
        }
        parameters.update(kwargs)

        super().__init__(parameters=parameters, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the TimeSeriesRegularizer.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: if self.time_index is None, if X and y have different lengths, if `time_index` in X does not
                        have an offset frequency that can be estimated
            TypeError: if the `time_index` column is not of type Datetime
            KeyError: if the `time_index` column doesn't exist
        """
        if self.time_index is None:
            raise ValueError("The argument time_index cannot be None!")
        elif self.time_index not in X.columns:
            raise KeyError(
                f"The time_index column `{self.time_index}` does not exist in X!",
            )

        X_ww = infer_feature_types(X)

        if not isinstance(X_ww.ww.logical_types[self.time_index], Datetime):
            raise TypeError(
                f"The time_index column `{self.time_index}` must be of type Datetime.",
            )

        if y is not None:
            y = infer_feature_types(y)
            if len(X_ww) != len(y):
                raise ValueError(
                    "If y has been passed, then it must be the same length as X.",
                )

        if self.frequency_payload:
            ww_payload = self.frequency_payload
        else:
            ww_payload = infer_frequency(
                X_ww[self.time_index],
                debug=True,
                window_length=self.window_length,
                threshold=self.threshold,
            )
        self.inferred_freq = ww_payload[0]
        self.debug_payload = ww_payload[1]

        if self.inferred_freq is not None:
            return self

        if (
            self.debug_payload["estimated_freq"] is None
        ):  # If even WW can't infer the frequency
            raise ValueError(
                f"The column {self.time_index} does not have a frequency that can be inferred.",
            )

        estimated_freq = self.debug_payload["estimated_freq"]
        duplicates = self.debug_payload["duplicate_values"]
        missing = self.debug_payload["missing_values"]
        extra = self.debug_payload["extra_values"]
        nan = self.debug_payload["nan_values"]

        self.error_dict = self._identify_indices(
            self.time_index,
            X_ww,
            estimated_freq,
            duplicates,
            missing,
            extra,
            nan,
        )

        return self

    @staticmethod
    def _identify_indices(
        time_index,
        X,
        estimated_freq,
        duplicates,
        missing,
        extra,
        nan,
    ):
        """Identifies which of the problematic indices is actually misaligned.

        Args:
            time_index (str): The column name of the datetime values to consider.
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            estimated_freq (str): The estimated frequency of the `time_index` column.
            duplicates (list): Payload information regarding the duplicate values.
            missing (list): Payload information regarding the missing values.
            extra (list): Payload information regarding the extra values.
            nan (list): Payload information regarding the nan values.

        Returns:
            (dict): A dictionary of the duplicate, missing, extra, and misaligned indices and their datetime values.
        """
        error_dict = {
            "duplicate": {},
            "missing": {},
            "extra": {},
            "nan": {},
            "misaligned": {},
        }

        # Adds the indices for the consecutive range of missing, duplicate, and extra values
        for each_missing in missing:
            # Needed to recreate what the missing datetime values would have been
            temp_dates = pd.date_range(
                pd.to_datetime(each_missing["dt"]),
                freq=estimated_freq,
                periods=each_missing["range"],
            )
            for each_range in range(each_missing["range"]):
                error_dict["missing"][each_missing["idx"] + each_range] = temp_dates[
                    each_range
                ]

        for each_duplicate in duplicates:
            for each_range in range(each_duplicate["range"]):
                error_dict["duplicate"][
                    each_duplicate["idx"] + each_range
                ] = pd.to_datetime(each_duplicate["dt"])

        for each_extra in extra:
            for each_range in range(each_extra["range"]):
                error_dict["extra"][each_extra["idx"] + each_range] = X.iloc[
                    each_extra["idx"] + each_range
                ][time_index]

        for each_nan in nan:
            for each_range in range(each_nan["range"]):
                error_dict["nan"][each_nan["idx"] + each_range] = "No Value"

        # Identify which of the duplicate/extra values in conjunction with the missing values are actually misaligned
        for ind_missing, missing_value in error_dict["missing"].items():
            temp_range = pd.date_range(missing_value, freq=estimated_freq, periods=3)
            window_range = temp_range[1] - temp_range[0]
            missing_range = [missing_value - window_range, missing_value + window_range]
            for ind_duplicate, duplicate_value in error_dict["duplicate"].items():
                if (
                    duplicate_value is not None
                    and missing_range[0] <= duplicate_value <= missing_range[1]
                ):
                    error_dict["misaligned"][ind_duplicate] = {
                        "incorrect": duplicate_value,
                        "correct": missing_value,
                    }
                    error_dict["duplicate"][ind_duplicate] = None
                    error_dict["missing"][ind_missing] = None
                    break
            for ind_extra, extra_value in error_dict["extra"].items():
                if (
                    extra_value is not None
                    and missing_range[0] <= extra_value <= missing_range[1]
                ):
                    error_dict["misaligned"][ind_extra] = {
                        "incorrect": extra_value,
                        "correct": missing_value,
                    }
                    error_dict["extra"][ind_extra] = None
                    error_dict["missing"][ind_missing] = None
                    break

        final_error_dict = {
            "duplicate": {},
            "missing": {},
            "extra": {},
            "nan": {},
            "misaligned": {},
        }
        # Remove duplicate/extra/missing values that were identified as misaligned
        for type_, type_inds in error_dict.items():
            new_type_inds = {
                ind_: date_ for ind_, date_ in type_inds.items() if date_ is not None
            }
            final_error_dict[type_] = new_type_inds

        return final_error_dict

    def transform(self, X, y=None):
        """Regularizes a dataframe and target data to an inferrable offset frequency.

        A 'clean' X and y (if y was passed in) are created based on an inferrable offset frequency and matching datetime values
        with the original X and y are imputed into the clean X and y. Datetime values identified as misaligned are
        shifted into their appropriate position.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            (pd.DataFrame, pd.Series): Data with an inferrable `time_index` offset frequency.
        """
        if self.inferred_freq is not None:
            return X, y

        # The cleaned df will begin at the range determined by estimated_range_start, which will result
        # in dropping of the first consecutive faulty values in the dataset.
        cleaned_df = pd.DataFrame(
            {
                self.time_index: pd.date_range(
                    self.debug_payload["estimated_range_start"],
                    self.debug_payload["estimated_range_end"],
                    freq=self.debug_payload["estimated_freq"],
                ),
            },
        )

        cleaned_x = cleaned_df.merge(X, on=[self.time_index], how="left")
        cleaned_x = cleaned_x.groupby(self.time_index).first().reset_index()

        cleaned_y = None
        if y is not None:
            y_dates = pd.DataFrame({self.time_index: X[self.time_index], "target": y})
            cleaned_y = cleaned_df.merge(y_dates, on=[self.time_index], how="left")
            cleaned_y = cleaned_y.groupby(self.time_index).first().reset_index()

        for index, values in self.error_dict["misaligned"].items():
            to_replace = X.iloc[index]
            to_replace[self.time_index] = values["correct"]
            cleaned_x.loc[
                cleaned_x[self.time_index] == values["correct"]
            ] = to_replace.values
            if y is not None:
                cleaned_y.loc[cleaned_y[self.time_index] == values["correct"]] = y.iloc[
                    index
                ]

        if cleaned_y is not None:
            cleaned_y = cleaned_y["target"]
            cleaned_y = ww.init_series(cleaned_y)

        cleaned_x.ww.init()

        return cleaned_x, cleaned_y
