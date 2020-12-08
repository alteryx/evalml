from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines import TimeSeriesBaselineRegressionPipeline


def test_time_series_baseline(ts_data):
    X, y = ts_data

    clf = TimeSeriesBaselineRegressionPipeline(parameters={"pipeline": {"gap": 1, "max_delay": 1}})
    clf.fit(X, y)

    np.testing.assert_allclose(clf.predict(X, y), y)


def test_time_series_baseline_no_X(ts_data):
    X, y = ts_data

    clf = TimeSeriesBaselineRegressionPipeline(parameters={"pipeline": {"gap": 1, "max_delay": 1}})
    clf.fit(X=None, y=y)

    np.testing.assert_allclose(clf.predict(X=None, y=y), y)


@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("gap,max_delay", [(0, 0), (1, 0), (0, 2), (1, 1), (1, 2), (2, 2), (7, 3), (2, 4)])
@patch("evalml.pipelines.RegressionPipeline._score_all_objectives")
def test_time_series_baseline_score_offset(mock_score, gap, max_delay, only_use_y, ts_data):
    X, y = ts_data

    expected_target = np.arange(1 + gap, 32)
    target_index = pd.date_range(f"2020-10-01", f"2020-10-{31-gap}")

    clf = TimeSeriesBaselineRegressionPipeline(parameters={"pipeline": {"gap": gap, "max_delay": max_delay}})

    if only_use_y:
        clf.fit(None, y)
        clf.score(X=None, y=y, objectives=[])
    else:
        clf.fit(X, y)
        clf.score(X, y, objectives=[])

    # Verify that NaNs are dropped before passed to objectives
    _, target, preds = mock_score.call_args[0]
    assert not target.isna().any()
    assert not preds.isna().any()

    # Target used for scoring matches expected dates
    pd.testing.assert_index_equal(target.index, target_index)
    np.testing.assert_equal(target.values, expected_target)
