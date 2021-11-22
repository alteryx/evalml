"""Transformer that delays input features and target variable for time series problems."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from skopt.space import Real
from statsmodels.tsa.stattools import acf
from woodwork import logical_types

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


class DelayedFeatureTransformer(Transformer):
    """Transformer that delays input features and target variable for time series problems.

    This component uses an algorithm based on the autocorrelation values of the target variable
    to determine which lags to select from the set of all possible lags.

    The algorithm is based on the idea that the local maxima of the autocorrelation function indicate the lags that have
    the most impact on the present time.

    The algorithm computes the autocorrelation values and finds the local maxima, called "peaks", that are significant at the given
    conf_level. Since lags in the range [0, 10] tend to be predictive but not local maxima, the union of the peaks is taken
    with the significant lags in the range [0, 10]. At the end, only selected lags in the range [0, max_delay] are used.

    Parametrizing the algorithm by conf_level lets the AutoMLAlgorithm tune the set of lags chosen so that the chances
    of finding a good set of lags is higher.

    Using conf_level value of 1 selects all possible lags.

    Args:
        date_index (str): Name of the column containing the datetime information used to order the data. Ignored.
        max_delay (int): Maximum number of time units to delay each feature. Defaults to 2.
        forecast_horizon (int): The number of time periods the pipeline is expected to forecast.
        conf_level (float): Float in range (0, 1] that determines the confidence interval size used to select
            which lags to compute from the set of [1, max_delay]. A delay of 1 will always be computed. If 1,
            selects all possible lags in the set of [1, max_delay], inclusive.
        delay_features (bool): Whether to delay the input features. Defaults to True.
        delay_target (bool): Whether to delay the target. Defaults to True.
        gap (int): The number of time units between when the features are collected and
            when the target is collected. For example, if you are predicting the next time step's target, gap=1.
            This is only needed because when gap=0, we need to be sure to start the lagging of the target variable
            at 1. Defaults to 1.
        random_seed (int): Seed for the random number generator. This transformer performs the same regardless of the random seed provided.
    """

    name = "Delayed Feature Transformer"
    hyperparameter_ranges = {"conf_level": Real(0.001, 1.0)}
    """{}"""
    needs_fitting = True
    target_colname_prefix = "target_delay_{}"
    """target_delay_{}"""

    def __init__(
        self,
        date_index=None,
        max_delay=2,
        gap=0,
        forecast_horizon=1,
        conf_level=0.05,
        delay_features=True,
        delay_target=True,
        random_seed=0,
        **kwargs,
    ):
        self.date_index = date_index
        self.max_delay = max_delay
        self.delay_features = delay_features
        self.delay_target = delay_target
        self.forecast_horizon = forecast_horizon
        self.gap = gap
        self.statistically_significant_lags = None

        if conf_level is None:
            raise ValueError("Parameter conf_level cannot be None.")

        if conf_level <= 0 or conf_level > 1:
            raise ValueError(
                f"Parameter conf_level must be in range (0, 1]. Received {conf_level}."
            )

        self.conf_level = conf_level

        self.start_delay = self.forecast_horizon + self.gap

        parameters = {
            "date_index": date_index,
            "max_delay": max_delay,
            "delay_target": delay_target,
            "delay_features": delay_features,
            "forecast_horizon": forecast_horizon,
            "conf_level": conf_level,
            "gap": gap,
        }
        parameters.update(kwargs)
        super().__init__(parameters=parameters, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits the DelayFeatureTransformer.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self

        Raises:
            ValueError: if self.date_index is None
        """
        if self.date_index is None:
            raise ValueError("date_index cannot be None!")
        self.statistically_significant_lags = self._find_significant_lags(
            y, conf_level=self.conf_level, max_delay=self.max_delay
        )
        return self

    @staticmethod
    def _encode_y_while_preserving_index(y):
        y_encoded = LabelEncoder().fit_transform(y)
        y = pd.Series(y_encoded, index=y.index)
        return y

    @staticmethod
    def _get_categorical_columns(X):
        return list(X.ww.select(["categorical", "boolean"], return_schema=True).columns)

    @staticmethod
    def _encode_X_while_preserving_index(X_categorical):
        return pd.DataFrame(
            OrdinalEncoder().fit_transform(X_categorical),
            columns=X_categorical.columns,
            index=X_categorical.index,
        )

    @staticmethod
    def _find_significant_lags(y, conf_level, max_delay):
        all_lags = np.arange(max_delay + 1)
        if y is not None:
            # Compute the acf and find its peaks
            acf_values, ci_intervals = acf(
                y, nlags=len(y) - 1, fft=True, alpha=conf_level
            )
            peaks, _ = find_peaks(acf_values)

            # Significant lags are the union of:
            # 1. the peaks (local maxima) that are significant
            # 2. The significant lags among the first 10 lags.
            # We then filter the list to be in the range [0, max_delay]
            index = np.arange(len(acf_values))
            significant = np.logical_or(ci_intervals[:, 0] > 0, ci_intervals[:, 1] < 0)
            first_significant_10 = index[:10][significant[:10]]
            significant_lags = (
                set(index[significant]).intersection(peaks).union(first_significant_10)
            )
            # If no lags are significant get the first lag
            significant_lags = sorted(significant_lags.intersection(all_lags)) or [1]
        else:
            significant_lags = all_lags
        return significant_lags

    def transform(self, X, y=None):
        """Computes the delayed features for all features in X and y.

        For each feature in X, it will add a column to the output dataframe for each
        delay in the (inclusive) range [1, max_delay]. The values of each delayed feature are simply the original
        feature shifted forward in time by the delay amount. For example, a delay of 3 units means that the feature
        value at row n will be taken from the n-3rd row of that feature

        If y is not None, it will also compute the delayed values for the target variable.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, or None): Target.

        Returns:
            pd.DataFrame: Transformed X.
        """
        # Normalize the data into pandas objects
        X_ww = infer_feature_types(X)
        cols_to_delay = list(
            X_ww.ww.select(
                ["numeric", "category", "boolean"], return_schema=True
            ).columns
        )
        X_ww = X_ww.ww.copy()
        categorical_columns = self._get_categorical_columns(X_ww)
        cols_derived_from_categoricals = []
        if self.delay_features and len(X) > 0:
            X_categorical = self._encode_X_while_preserving_index(
                X_ww[categorical_columns]
            )
            for col_name in cols_to_delay:
                col = X_ww[col_name]
                if col_name in categorical_columns:
                    col = X_categorical[col_name]
                for t in self.statistically_significant_lags:
                    feature_name = f"{col_name}_delay_{self.start_delay + t}"
                    X_ww.ww[f"{col_name}_delay_{self.start_delay + t}"] = col.shift(
                        self.start_delay + t
                    )
                    if col_name in categorical_columns:
                        cols_derived_from_categoricals.append(feature_name)
        # Handle cases where the target was passed in
        if self.delay_target and y is not None:
            y = infer_feature_types(y)
            if type(y.ww.logical_type) == logical_types.Categorical:
                y = self._encode_y_while_preserving_index(y)
            for t in self.statistically_significant_lags:
                X_ww.ww[
                    self.target_colname_prefix.format(t + self.start_delay)
                ] = y.shift(self.start_delay + t)
        # Features created from categorical columns should no longer be categorical
        X_ww.ww.set_types({col: "Double" for col in cols_derived_from_categoricals})
        return X_ww.ww.drop(cols_to_delay)

    def fit_transform(self, X, y):
        """Fit the component and transform the input data.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, or None): Target.

        Returns:
            pd.DataFrame: Transformed X.
        """
        return self.fit(X, y).transform(X, y)
