"""Time series estimator that predicts using the naive forecasting approach."""
import numpy as np

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.pipelines.components.transformers import DelayedFeatureTransformer
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class TimeSeriesBaselineEstimator(Estimator):
    """Time series estimator that predicts using the naive forecasting approach.

    This is useful as a simple baseline estimator for time series problems.

    Args:
        gap (int): Gap between prediction date and target date and must be a positive integer. If gap is 0, target date will be shifted ahead by 1 time period. Defaults to 1.
        forecast_horizon (int): Number of time steps the model is expected to predict.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Time Series Baseline Estimator"
    hyperparameter_ranges = {}
    """{}"""
    model_family = ModelFamily.BASELINE
    """ModelFamily.BASELINE"""
    supported_problem_types = [
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""
    predict_uses_y = False

    def __init__(self, gap=1, forecast_horizon=1, random_seed=0, **kwargs):
        self._prediction_value = None
        self.start_delay = forecast_horizon + gap
        self._classes = None

        if gap < 0:
            raise ValueError(
                f"gap value must be a positive integer. {gap} was provided."
            )

        parameters = {"gap": gap, "forecast_horizon": forecast_horizon}
        parameters.update(kwargs)
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fits time series baseline estimator to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If input y is None.
        """
        if y is None:
            raise ValueError("Cannot fit Time Series Baseline Classifier if y is None")
        vals, _ = np.unique(y, return_counts=True)
        self._classes = list(vals)
        return self

    def predict(self, X):
        """Make predictions using fitted time series baseline estimator.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If input y is None.
        """
        X = infer_feature_types(X)
        feature_name = DelayedFeatureTransformer.target_colname_prefix.format(
            self.start_delay
        )
        if feature_name not in X.columns:
            raise ValueError(
                "Time Series Baseline Estimator is meant to be used in a pipeline with "
                "a DelayedFeaturesTransformer"
            )
        return X.ww[feature_name]

    def predict_proba(self, X):
        """Make prediction probabilities using fitted time series baseline estimator.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.DataFrame: Predicted probability values.

        Raises:
            ValueError: If input y is None.
        """
        preds = self.predict(X).astype("int")
        proba_arr = np.zeros((len(preds), len(self._classes)))
        proba_arr[np.arange(len(preds)), preds] = 1
        return infer_feature_types(proba_arr)

    @property
    def feature_importance(self):
        """Returns importance associated with each feature.

        Since baseline estimators do not use input features to calculate predictions, returns an array of zeroes.

        Returns:
            np.ndarray (float): An array of zeroes.
        """
        return np.zeros(1)
