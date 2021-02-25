import numpy as np
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

    def __init__(self, date_column='ds', p=0, d=0, q=0,
                 random_state=0, **kwargs):
        self.date_column = date_column

        parameters = {'p': p,
                      "d": d,
                      "q": q}

        parameters.update(kwargs)

        p_error_msg = "ARIMA is not installed. Please install using `pip install statsmodels`."
        arima = import_or_raise("statsmodels.tsa.arima_model", error_msg=p_error_msg)
        arima_regressor = arima.ARIMA(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=arima_regressor,
                         random_state=random_state)

    def fit(self, X, y=None):
        p_error_msg = "ARIMA is not installed. Please install using `pip install statsmodels`."
        arima = import_or_raise("statsmodels.tsa.arima_model", error_msg=p_error_msg)
        X, y = self._manage_woodwork(X, y)

        if y is None:
            raise ValueError('ARIMA Regressor requires y as input.')
        elif X:
            arima_with_data = arima.ARIMA(endog=y, exog=X, **self.parameters)
        else:
            arima_with_data = arima.ARIMA(endog=y, **self.parameters)

        self._component_obj = arima_with_data
        self._component_obj.fit()
        return self

    def predict(self, X, y=None):
        X, y = self._manage_woodwork(X, y)
        if y is None:
            raise ValueError('ARIMA Regressor requires y as input.')
        elif X:
            y_pred = self._component_obj.predict(endog=y, exog=X)
        else:
            y_pred = self._component_obj.predict(endog=y)
        return y_pred

    @property
    def feature_importance(self):
        """
        Returns array of 0's with len(1) as feature_importance is not defined for ARIMA regressor.
        """
        return np.zeros(1)
