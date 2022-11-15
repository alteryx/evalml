"""Transformer that delays input features and target variable for time series problems."""
import numpy as np
import pandas as pd
import woodwork as ww
from featuretools.primitives import RollingMean
from scipy.signal import find_peaks
from sklearn.preprocessing import OrdinalEncoder
from skopt.space import Real
from statsmodels.tsa.stattools import acf
from woodwork import logical_types

from evalml.pipelines.components.transformers import LabelEncoder
from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types


class TimeSeriesFeaturizer(Transformer):
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
        time_index (str): Name of the column containing the datetime information used to order the data. Ignored.
        max_delay (int): Maximum number of time units to delay each feature. Defaults to 2.
        forecast_horizon (int): The number of time periods the pipeline is expected to forecast.
        conf_level (float): Float in range (0, 1] that determines the confidence interval size used to select
            which lags to compute from the set of [1, max_delay]. A delay of 1 will always be computed. If 1,
            selects all possible lags in the set of [1, max_delay], inclusive.
        rolling_window_size (float): Float in range (0, 1] that determines the size of the window used for rolling
            features. Size is computed as rolling_window_size * max_delay.
        delay_features (bool): Whether to delay the input features. Defaults to True.
        delay_target (bool): Whether to delay the target. Defaults to True.
        gap (int): The number of time units between when the features are collected and
            when the target is collected. For example, if you are predicting the next time step's target, gap=1.
            This is only needed because when gap=0, we need to be sure to start the lagging of the target variable
            at 1. Defaults to 1.
        random_seed (int): Seed for the random number generator. This transformer performs the same regardless of the random seed provided.
    """

    name = "Time Series Featurizer"
    hyperparameter_ranges = {
        "conf_level": Real(0.001, 1.0),
        "rolling_window_size": Real(0.001, 1.0),
    }
    """{"conf_level": Real(0.001, 1.0),
        "rolling_window_size": Real(0.001, 1.0)
    }"""
    needs_fitting = True
    target_colname_prefix = "target_delay_{}"
    """target_delay_{}"""

    def __init__(
        self,
        time_index=None,
        max_delay=2,
        gap=0,
        forecast_horizon=1,
        conf_level=0.05,
        rolling_window_size=0.25,
        delay_features=True,
        delay_target=True,
        random_seed=0,
        **kwargs,
    ):
        self.time_index = time_index
        self.max_delay = max_delay
        self.delay_features = delay_features
        self.delay_target = delay_target
        self.forecast_horizon = forecast_horizon
        self.gap = gap
        self.rolling_window_size = rolling_window_size
        self.statistically_significant_lags = None

        if conf_level is None:
            raise ValueError("Parameter conf_level cannot be None.")

        if conf_level <= 0 or conf_level > 1:
            raise ValueError(
                f"Parameter conf_level must be in range (0, 1]. Received {conf_level}.",
            )

        self.conf_level = conf_level

        self.start_delay = self.forecast_horizon + self.gap

        parameters = {
            "time_index": time_index,
            "max_delay": max_delay,
            "delay_target": delay_target,
            "delay_features": delay_features,
            "forecast_horizon": forecast_horizon,
            "conf_level": conf_level,
            "gap": gap,
            "rolling_window_size": rolling_window_size,
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
            ValueError: if self.time_index is None
        """
        if self.time_index is None:
            raise ValueError("time_index cannot be None!")
        self.statistically_significant_lags = self._find_significant_lags(
            y,
            conf_level=self.conf_level,
            start_delay=self.start_delay,
            max_delay=self.max_delay,
        )
        return self

    @staticmethod
    def _encode_y_while_preserving_index(y):
        y_encoded = LabelEncoder().fit_transform(None, y)[1]
        y = pd.Series(y_encoded, index=y.index)
        return y

    @staticmethod
    def _get_categorical_columns(X):
        return list(X.ww.select(["category", "boolean"], return_schema=True).columns)

    @staticmethod
    def _encode_X_while_preserving_index(X_categorical):
        return pd.DataFrame(
            OrdinalEncoder().fit_transform(X_categorical),
            columns=X_categorical.columns,
            index=X_categorical.index,
        )

    @staticmethod
    def _find_significant_lags(y, conf_level, start_delay, max_delay):
        all_lags = np.arange(start_delay, start_delay + max_delay + 1)
        if y is not None:
            # Compute the acf and find its peaks
            acf_values, ci_intervals = acf(
                y,
                nlags=len(y) - 1,
                fft=True,
                alpha=conf_level,
            )
            peaks, _ = find_peaks(acf_values)
            # Significant lags are the union of:
            # 1. the peaks (local maxima) that are significant
            # 2. The significant lags among the first 10 lags.
            # We then filter the list to be in the range [start_delay, start_delay + max_delay]
            index = np.arange(len(acf_values))
            significant = np.logical_or(ci_intervals[:, 0] > 0, ci_intervals[:, 1] < 0)
            first_significant_10 = index[:10][significant[:10]]
            significant_lags = (
                set(index[significant]).intersection(peaks).union(first_significant_10)
            )
            # If no lags are significant get the first lag
            significant_lags = sorted(significant_lags.intersection(all_lags)) or [
                start_delay,
            ]
        else:
            significant_lags = all_lags
        return significant_lags

    def _compute_rolling_transforms(self, X, y, original_features):
        """Compute the rolling features from the original features.

        Args:
            X (pd.DataFrame or None): Data to transform.
            y (pd.Series, or None): Target.

        Returns:
            pd.DataFrame: Data with rolling features. All new features.
        """
        size = int(self.rolling_window_size * self.max_delay)
        rolling_mean = RollingMean(
            window_length=size + 1,
            gap=self.start_delay,
            min_periods=size + 1,
        )
        rolling_mean = rolling_mean.get_function()
        numerics = sorted(
            set(X.ww.select(["numeric"], return_schema=True).columns).intersection(
                original_features,
            ),
        )

        data = pd.DataFrame(
            {f"{col}_rolling_mean": rolling_mean(X.index, X[col]) for col in numerics},
        )
        if y is not None and "numeric" in y.ww.semantic_tags:
            data[f"target_rolling_mean"] = rolling_mean(y.index, y)
        data.index = X.index
        data.ww.init(
            logical_types={col: "Double" for col in data.columns},
        )
        return data

    def _compute_delays(self, X_ww, y):
        """Computes the delayed features for numeric/categorical features in X and y.

        Use the autocorrelation to determine delays.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, or None): Target.

        Returns:
            pd.DataFrame: Data with original features and delays.
        """
        cols_to_delay = list(
            X_ww.ww.select(
                ["numeric", "category", "boolean"],
                return_schema=True,
            ).columns,
        )
        categorical_columns = self._get_categorical_columns(X_ww)
        cols_derived_from_categoricals = []
        lagged_features = {}
        if self.delay_features and len(X_ww) > 0:
            X_categorical = self._encode_X_while_preserving_index(
                X_ww[categorical_columns],
            )
            for col_name in cols_to_delay:

                col = X_ww[col_name]
                if col_name in categorical_columns:
                    col = X_categorical[col_name]
                for t in self.statistically_significant_lags:
                    feature_name = f"{col_name}_delay_{t}"
                    lagged_features[f"{col_name}_delay_{t}"] = col.shift(t)
                    if col_name in categorical_columns:
                        cols_derived_from_categoricals.append(feature_name)
        # Handle cases where the target was passed in
        if self.delay_target and y is not None:
            if type(y.ww.logical_type) == logical_types.Categorical:
                y = self._encode_y_while_preserving_index(y)
            for t in self.statistically_significant_lags:
                lagged_features[self.target_colname_prefix.format(t)] = y.shift(t)
        # Features created from categorical columns should no longer be categorical
        lagged_features = pd.DataFrame(lagged_features)
        lagged_features.ww.init(
            logical_types={col: "Double" for col in lagged_features.columns},
        )
        lagged_features.index = X_ww.index
        return ww.concat_columns([X_ww, lagged_features])

    def transform(self, X, y=None):
        """Computes the delayed values and rolling means for X and y.

        The chosen delays are determined by the autocorrelation function of the target variable. See the class docstring
        for more information on how they are chosen. If y is None, all possible lags are chosen.

        If y is not None, it will also compute the delayed values for the target variable.

        The rolling means for all numeric features in X and y, if y is numeric, are also returned.

        Args:
            X (pd.DataFrame or None): Data to transform. None is expected when only the target variable is being used.
            y (pd.Series, or None): Target.

        Returns:
            pd.DataFrame: Transformed X. No original features are returned.
        """
        if y is not None:
            y = infer_feature_types(y)
        # Normalize the data into pandas objects
        X_ww = infer_feature_types(X)
        original_features = [col for col in X_ww.columns if col != self.time_index]
        delayed_features = self._compute_delays(X_ww, y)
        rolling_means = self._compute_rolling_transforms(X_ww, y, original_features)
        features = ww.concat_columns([delayed_features, rolling_means])
        return features.ww.drop(original_features)

    def fit_transform(self, X, y=None):
        """Fit the component and transform the input data.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, or None): Target.

        Returns:
            pd.DataFrame: Transformed X.
        """
        return self.fit(X, y).transform(X, y)
