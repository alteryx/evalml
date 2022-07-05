"""Component that removes trends from time series by fitting a polynomial to the data."""
import pandas as pd
from skopt.space import Integer
from sktime.forecasting.base._fh import ForecastingHorizon
from statsmodels.tsa.seasonal import seasonal_decompose

from evalml.pipelines.components.transformers.preprocessing import Detrender
from evalml.utils import import_or_raise, infer_feature_types


class PolynomialDetrender(Detrender):
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
                    f"Parameter Degree must be an integer!: Received {type(degree).__name__}",
                )

        params = {"degree": degree}
        params.update(kwargs)
        error_msg = "sktime is not installed. Please install using 'pip install sktime'"

        trend = import_or_raise("sktime.forecasting.trend", error_msg=error_msg)
        detrend = import_or_raise(
            "sktime.transformations.series.detrend",
            error_msg=error_msg,
        )

        detrender = detrend.Detrender(trend.PolynomialTrendForecaster(degree=degree))

        super().__init__(
            parameters=params,
            component_obj=detrender,
            random_seed=random_seed,
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
        y_t = pd.Series(y_t, index=y_ww.index)
        y_t.ww.init(logical_type="double")
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
        y_ww = infer_feature_types(y)
        y_t = self._component_obj.inverse_transform(y_ww)
        y_t = infer_feature_types(pd.Series(y_t, index=y_ww.index))
        return y_t

    def get_trend_dataframe(self, X, y):
        """Return a list of dataframes with 3 columns: trend, seasonality, residual.

        Scikit-learn's PolynomialForecaster is used to generate the trend portion of the target data. statsmodel's
        seasonal_decompose is used to generate the seasonality of the data.

        Args:
            X (pd.DataFrame): Input data with time series data in index.
            y (pd.Series or pd.DataFrame): Target variable data provided as a Series for univariate problems or
                a DataFrame for multivariate problems.

        Returns:
            list of pd.DataFrame: Each DataFrame contains the columns "trend", "seasonality" and "residual,"
                with the column values being the decomposed elements of the target data.

        Raises:
            TypeError: If X does not have time-series data in the index.
            ValueError: If time series index of X does not have an inferred frequency.
            TypeError: If y is not provided as a pandas Series or DataFrame.

        """
        X = infer_feature_types(X)
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("Provided X should have datetimes in the index.")
        if X.index.freq is None:
            raise ValueError(
                "Provided DatetimeIndex of X should have an inferred frequency."
            )
        fh = ForecastingHorizon(X.index, is_relative=False)

        result_dfs = []

        def _decompose_target(X, y, fh):
            """Function to generate a single DataFrame with trend, seasonality and residual components."""
            forecaster = self._component_obj.forecaster.clone()
            forecaster.fit(y=y, X=X)
            trend = forecaster.predict(fh=fh, X=y)
            seasonality = seasonal_decompose(y).seasonal
            residual = y - trend - seasonality
            return pd.DataFrame(
                {
                    "trend": trend,
                    "seasonality": seasonality,
                    "residual": residual,
                }
            )

        if isinstance(y, pd.Series):
            result_dfs.append(_decompose_target(X, y, fh))
        elif isinstance(y, pd.DataFrame):
            for colname in y.columns:
                result_dfs.append(_decompose_target(X, y[colname], fh))
        else:
            raise TypeError("y must be pd.Series or pd.DataFrame!")

        return result_dfs
