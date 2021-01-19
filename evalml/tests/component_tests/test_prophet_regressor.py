import pandas as pd
from fbprophet import Prophet

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ProphetRegressor
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import suppress_stdout_stderr


def test_model_family():
    assert ProphetRegressor.model_family == ModelFamily.PROPHET


def test_problem_types():
    assert set(ProphetRegressor.supported_problem_types) == {ProblemTypes.TIME_SERIES_REGRESSION}


def test_fit_predict_ts(ts_data):
    X, y = ts_data

    def build_prophet_df(X, y=None):
        # check for datetime column
        if 'ds' in X.columns:
            date_col = X['ds']
        elif isinstance(X.index, pd.DatetimeIndex):
            date_col = X.reset_index()
            date_col = date_col['index']
        else:
            date_col = X.select_dtypes(include='datetime')
            if date_col.shape[1] == 0:
                raise ValueError('Prophet estimator requires input data X to have a datetime column')

        date_col = date_col.rename('ds')
        prophet_df = date_col.to_frame()
        if y is not None:
            y.index = prophet_df.index
            prophet_df['y'] = y
        return prophet_df

    p_clf = Prophet()
    prophet_df = build_prophet_df(X, y)

    with suppress_stdout_stderr():
        p_clf.fit(prophet_df)
    y_pred_p = p_clf.predict(prophet_df)['yhat']

    clf = ProphetRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    assert (y_pred == y_pred_p).all()
