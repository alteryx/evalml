"""Transformer that regularizes a dataset with an uninferrable offset frequency for time series problems."""
import pandas as pd
from woodwork.logical_types import Datetime
from woodwork.statistics_utils import infer_frequency

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


def _realign_dates(cleaned_x, cleaned_y, X, y, time_index, issue_dates, error_dict):
    """Realigns observations whose datetime values have been identified as misaligned.

    Args:
        cleaned_x (pd.DataFrame): The expected 'clean' training data.
        cleaned_y (pd.Series, optional): The expected 'clean' target training data.
        X (pd.DataFrame): The actual input training data of shape.
        y (pd.Series): The actual target training data.
        time_index (str): The column indicating the time index.
        issue_dates (dict): Unmatched datetime values.
        error_dict (dict): Dictionary of all faulty datetime values.

    Returns:
        (pd.DataFrame, pd.Series): Data with an inferrable `time_index` offset frequency with realigned observations.
    """
    misaligned_ind_clean = {}

    for issue_ind, issue_val in issue_dates.items():
        if issue_ind not in error_dict["missing"].keys():
            misaligned_ind_clean[issue_ind] = issue_val

    for misaligned_ind_original, each_misaligned_original in error_dict[
        "misaligned"
    ].items():
        clean_x_date = pd.to_datetime(each_misaligned_original["correct"])
        feature_values = X.loc[
            X.index == misaligned_ind_original,
            list(set(cleaned_x.columns) - {time_index}),
        ].iloc[0]
        cleaned_x.loc[
            cleaned_x[time_index] == clean_x_date,
            list(set(cleaned_x.columns) - {time_index}),
        ] = feature_values.values

        if cleaned_y is not None:
            for (
                misaligned_clean_ind,
                misaligned_clean_val,
            ) in misaligned_ind_clean.items():
                if pd.to_datetime(misaligned_clean_val) == clean_x_date:
                    cleaned_y.iloc[misaligned_clean_ind] = y.iloc[
                        misaligned_ind_original
                    ]
    return cleaned_x, cleaned_y


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
        time_index (string): Name of the column containing the datetime information used to order the data.
        random_seed (int): Seed for the random number generator. This transformer performs the same regardless of the random seed provided.
    """

    name = "Time Series Regularizer"
    hyperparameter_ranges = {}
    """{}"""

    modifies_target = True
    training_only = True

    def __init__(
        self, time_index=None, window_length=5, threshold=0.8, random_seed=0, **kwargs
    ):
        self.time_index = time_index
        self.window_length = window_length
        self.threshold = threshold
        self.error_dict = {}
        self.inferred_freq = None
        self.debug_payload = None

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
            ValueError: if self.time_index is None
            TypeError: if the `time_index` column is not of type Datetime
            ValueError: if X and y have different lengths
            ValueError: if `time_index` in X does not have an offset frequency that can be estimated
        """
        if self.time_index is None:
            raise ValueError("The argument time_index cannot be None!")

        X_ww = infer_feature_types(X)

        if not isinstance(X_ww.ww.logical_types[self.time_index], Datetime):
            raise TypeError(
                f"The time_index column {self.time_index} must be of type Datetime."
            )

        if y is not None:
            y = infer_feature_types(y)
            if len(X_ww) != len(y):
                raise ValueError(
                    "If y has been passed, then it must be the same length as X."
                )

        ww_payload = infer_frequency(
            X[self.time_index],
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
                f"The column {self.time_index} does not have a frequency that can be inferred."
            )

        estimated_freq = self.debug_payload["estimated_freq"]
        duplicates = self.debug_payload["duplicate_values"]
        missing = self.debug_payload["missing_values"]
        extra = self.debug_payload["extra_values"]
        nan = self.debug_payload["nan_values"]

        self.error_dict = self._identify_indices(
            self.time_index, X, estimated_freq, duplicates, missing, extra, nan
        )

        return self

    @staticmethod
    def _identify_indices(
        time_index, X, estimated_freq, duplicates, missing, extra, nan
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

    def create_clean_x_y(self, X, y):
        """Creates a clean X and y to repopulate with the original X and y values based off of an inferrable frequency.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            (pd.DataFrame, pd.Series, dict): X and y based off of an inferrable `time_index` range and a dict of
            unmatched datetime values.
        """
        clean_x = pd.DataFrame(columns=X.columns)

        expected_ts = pd.date_range(
            start=self.debug_payload["estimated_range_start"],
            end=self.debug_payload["estimated_range_end"],
            freq=self.debug_payload["estimated_freq"],
        )

        issue_dates = {}

        clean_x[self.time_index] = expected_ts
        clean_y = pd.Series([None] * len(clean_x)) if y is not None else None

        for ind_, datetime in enumerate(clean_x[self.time_index]):
            if pd.Timestamp(datetime) in X[self.time_index].values:
                # If the datetime value in the clean dataset is in X more than once, then select only the features of
                # the first of these duplicate observations
                feature_values = X.loc[
                    X[self.time_index] == pd.Timestamp(datetime)
                ].iloc[0]
                clean_x.iloc[ind_] = feature_values.values
                if clean_y is not None:
                    clean_y.iloc[ind_] = y.iloc[feature_values.name]
            else:
                clean_x.loc[
                    clean_x[self.time_index] == datetime,
                    list(set(clean_x.columns) - {self.time_index}),
                ] = None
                issue_dates[ind_] = datetime
        return clean_x, clean_y, issue_dates

    def transform(self, X, y=None):
        """Regularizes a dataframe and target data to an inferrable offset frequency.

        A 'clean' X and y (if passed) are created based on an inferrable offset frequency and matching datetime values
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

        X_ww = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)

        cleaned_x, cleaned_y, issue_dates = self.create_clean_x_y(X_ww, y)
        cleaned_x, cleaned_y = _realign_dates(
            cleaned_x, cleaned_y, X_ww, y, self.time_index, issue_dates, self.error_dict
        )
        return cleaned_x, cleaned_y
