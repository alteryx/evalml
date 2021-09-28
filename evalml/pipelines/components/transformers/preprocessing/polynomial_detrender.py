"""Component that removes trends from time series by fitting a polynomial to the data."""
import pandas as pd
from skopt.space import Integer

from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import import_or_raise, infer_feature_types


class PolynomialDetrender(Transformer):
    """Removes trends from time series by fitting a polynomial to the data.

    Args:
        degree (int): Degree for the polynomial. If 1, linear model is fit to the data.
            If 2, quadratic model is fit, etc. Defaults to 1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Polynomial Detrender"
    hyperparameter_ranges = {"degree": Integer(1, 3)}
    """{
        "degree": Integer(1, 3)
    }"""
    modifies_features = False
    modifies_target = True

    def __init__(self, degree=1, random_seed=0, **kwargs):
        if not isinstance(degree, int):
            if isinstance(degree, float) and degree.is_integer():
                degree = int(degree)
            else:
                raise TypeError(
                    f"Parameter Degree must be an integer!: Received {type(degree).__name__}"
                )

        params = {"degree": degree}
        params.update(kwargs)
        error_msg = "sktime is not installed. Please install using 'pip install sktime'"

        trend = import_or_raise("sktime.forecasting.trend", error_msg=error_msg)
        detrend = import_or_raise(
            "sktime.transformations.series.detrend", error_msg=error_msg
        )

        detrender = detrend.Detrender(trend.PolynomialTrendForecaster(degree=degree))

        super().__init__(
            parameters=params, component_obj=detrender, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fits the PolynomialDetrender.

        Args:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to detrend.

        Returns:
            self

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise ValueError("y cannot be None for PolynomialDetrender!")
        y_dt = infer_feature_types(y)
        self._component_obj.fit(y_dt)
        return self

    def transform(self, X, y=None):
        """Removes fitted trend from target variable.

        Args:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to detrend.

        Returns:
            tuple of pd.DataFrame, pd.Series: The input features are returned without modification. The target
                variable y is detrended
        """
        if y is None:
            return X, y
        y_ww = infer_feature_types(y)
        y_t = self._component_obj.transform(y_ww)
        y_t = infer_feature_types(pd.Series(y_t, index=y_ww.index))
        return X, y_t

    def fit_transform(self, X, y=None):
        """Removes fitted trend from target variable.

        Args:
            X (pd.DataFrame, optional): Ignored.
            y (pd.Series): Target variable to detrend.

        Returns:
            tuple of pd.DataFrame, pd.Series: The first element are the input features returned without modification.
                The second element is the target variable y with the fitted trend removed.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y):
        """Adds back fitted trend to target variable.

        Args:
            y (pd.Series): Target variable.

        Returns:
            tuple of pd.DataFrame, pd.Series: The first element are the input features returned without modification.
                The second element is the target variable y with the trend added back.

        Raises:
            ValueError: If y is None.
        """
        if y is None:
            raise ValueError("y cannot be None for PolynomialDetrender!")
        y_dt = infer_feature_types(y)
        y_t = self._component_obj.inverse_transform(y_dt)
        y_t = infer_feature_types(pd.Series(y_t, index=y_dt.index))
        return y_t
