import numpy as np

from evalml.pipelines import TimeSeriesBaselineRegressionPipeline


def test_time_series_baseline(ts_data):
    X, y = ts_data

    clf = TimeSeriesBaselineRegressionPipeline(parameters={"pipeline": {"gap": 1, "max_delay": 1}})
    clf.fit(X, y)

    np.testing.assert_allclose(clf.predict(X, y), y)


def test_time_series_baseline_no_X(ts_data):
    X, y = ts_data

    clf = TimeSeriesBaselineRegressionPipeline(parameters={"pipeline": {"gap": 1, "max_delay": 1}})
    clf.fit(X, y=y)

    np.testing.assert_allclose(clf.predict(X=None, y=y), y)
