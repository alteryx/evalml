import numpy as np

from evalml.pipelines import TimeSeriesBaselineRegressionPipeline


def test_time_series_baseline(ts_data):
    X, y = ts_data

    first = y.iloc[0]
    y_true = y.shift(periods=1)
    y_true.iloc[0] = first
    clf = TimeSeriesBaselineRegressionPipeline(parameters={"pipeline": {"gap": 0, "max_delay": 0}})
    clf.fit(X, y)

    np.testing.assert_allclose(clf.predict(X, y), y_true)
