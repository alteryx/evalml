import numpy as np
import pandas as pd
from skopt.space import Integer

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise


class ARIMARegressor(Estimator):
    """
    Autoregressive Integrated Moving Average Model.
    The three parameters (p, d, q) are the AR order, the degree of differencing, and the MA order.
    More information here: https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMA.html

    """
    name = "ARIMA Regressor"
    hyperparameter_ranges = {
        "p": Integer(0, 10),
        "d": Integer(0, 10),
        "q": Integer(0, 10),
    }
    model_family = ModelFamily.ARIMA
    supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]

    def __init__(self, date_column='date', p=0, d=0, q=0,
                 random_seed=0, **kwargs):
        self.date_column = date_column

        order = (p, d, q)
        parameters = {'order': order}
        parameters.update(kwargs)

        p_error_msg = "ARIMA is not installed. Please install using `pip install statsmodels`."
        import_or_raise("statsmodels.tsa.arima_model", error_msg=p_error_msg)
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def get_dates(self, X, y):
        if isinstance(y.index, pd.DatetimeIndex):
            date_col = y.index
        elif X is not None:
            if self.date_column in X.columns:
                date_col = X[self.date_column]
            elif isinstance(X.index, pd.DatetimeIndex):
                date_col = X.index
        else:
            msg = "ARIMA regressor requires input data X to have a datetime column specified by the 'date_column' parameter. If not it will look for the datetime column in the index of X or y."
            raise ValueError(msg)
        return date_col

    def fit(self, X, y=None):
        p_error_msg = "ARIMA is not installed. Please install using `pip install statsmodels`."
        arima = import_or_raise("statsmodels.tsa.arima_model", error_msg=p_error_msg)
        X, y = self._manage_woodwork(X, y)

        dates = self.get_dates(X, y)
        if y is None:
            raise ValueError('ARIMA Regressor requires y as input.')
        elif X is not None:
            arima_with_data = arima.ARIMA(endog=y, exog=X, dates=dates, **self.parameters)
        else:
            arima_with_data = arima.ARIMA(endog=y, dates=dates, **self.parameters)

        self._component_obj = arima_with_data
        self._component_obj.fit(solver='nm')
        return self

    def predict(self, X, y=None):
        X, y = self._manage_woodwork(X, y)
        dates = self.get_dates(X, y)
        start = dates.min()
        end = dates.max()
        params = self.parameters['order']
        if X:
            y_pred = self._component_obj.predict(params=params, start=start, end=end, exog=X)
        else:
            y_pred = self._component_obj.predict(params=params, start=start, end=end)
        return y_pred

    @property
    def feature_importance(self):
        """
        Returns array of 0's with len(1) as feature_importance is not defined for ARIMA regressor.
        """
        return np.zeros(1)
