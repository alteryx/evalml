from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.pipelines import TimeSeriesBaselineRegressionPipeline
from evalml.pipelines.time_series_baselines import (
    TimeSeriesBaselineBinaryPipeline,
    TimeSeriesBaselineMulticlassPipeline
)


@pytest.mark.parametrize('X_none', [True, False])
@pytest.mark.parametrize('gap', [0, 1])
@pytest.mark.parametrize('pipeline_class', [TimeSeriesBaselineRegressionPipeline,
                                            TimeSeriesBaselineBinaryPipeline, TimeSeriesBaselineMulticlassPipeline])
@patch("evalml.pipelines.TimeSeriesClassificationPipeline._decode_targets", side_effect=lambda y: y)
def test_time_series_baseline(mock_decode, pipeline_class, gap, X_none, ts_data):
    X, y = ts_data

    clf = pipeline_class(parameters={"pipeline": {"gap": gap, "max_delay": 1},
                                     "Time Series Baseline Estimator": {'gap': gap, 'max_delay': 1}})
    expected_y = y.shift(1) if gap == 0 else y
    expected_y = expected_y.reset_index(drop=True)
    if not expected_y.isnull().values.any():
        expected_y = expected_y.astype("Int64")
    if X_none:
        X = None
    clf.fit(X, y)
    assert_series_equal(expected_y, clf.predict(X, y).to_series())


@pytest.mark.parametrize('X_none', [True, False])
@pytest.mark.parametrize('gap', [0, 1])
@pytest.mark.parametrize('pipeline_class', [TimeSeriesBaselineBinaryPipeline, TimeSeriesBaselineMulticlassPipeline])
def test_time_series_baseline_predict_proba(pipeline_class, gap, X_none):
    X = pd.DataFrame({"a": [4, 5, 6, 7, 8]})
    y = pd.Series([0, 1, 1, 0, 1])
    expected_proba = pd.DataFrame({0: pd.Series([1, 0, 0, 1, 0], dtype="float64"),
                                   1: pd.Series([0, 1, 1, 0, 1], dtype="float64")})
    if pipeline_class == TimeSeriesBaselineMulticlassPipeline:
        y = pd.Series([0, 1, 2, 2, 1])
        expected_proba = pd.DataFrame({0: pd.Series([1, 0, 0, 0, 0], dtype="float64"),
                                       1: pd.Series([0, 1, 0, 0, 1], dtype="float64"),
                                       2: pd.Series([0, 0, 1, 1, 0], dtype="float64")})
    if gap == 0:
        # Shift to pad the first row with Nans
        expected_proba = expected_proba.shift(1)

    clf = pipeline_class(parameters={"pipeline": {"gap": gap, "max_delay": 1},
                                     "Time Series Baseline Estimator": {'gap': gap, 'max_delay': 1}})
    if X_none:
        X = None
    clf.fit(X, y)
    assert_frame_equal(expected_proba, clf.predict_proba(X, y).to_dataframe())


@pytest.mark.parametrize('pipeline_class', [TimeSeriesBaselineRegressionPipeline,
                                            TimeSeriesBaselineBinaryPipeline, TimeSeriesBaselineMulticlassPipeline])
@pytest.mark.parametrize("only_use_y", [True, False])
@pytest.mark.parametrize("gap,max_delay", [(0, 0), (1, 0), (0, 2), (1, 1), (1, 2), (2, 2), (7, 3), (2, 4)])
@patch("evalml.pipelines.RegressionPipeline._score_all_objectives")
@patch("evalml.pipelines.ClassificationPipeline._score_all_objectives")
@patch("evalml.pipelines.ClassificationPipeline._encode_targets", side_effect=lambda y: y)
def test_time_series_baseline_score_offset(mock_encode, mock_classification_score, mock_regression_score, gap, max_delay,
                                           only_use_y, pipeline_class, ts_data):
    X, y = ts_data

    expected_target = pd.Series(np.arange(1 + gap, 32), index=pd.date_range(f"2020-10-01", f"2020-10-{31-gap}"))
    if gap == 0:
        expected_target = expected_target[1:]
    clf = pipeline_class(parameters={"pipeline": {"gap": gap, "max_delay": max_delay},
                                     "Time Series Baseline Estimator": {"gap": gap, "max_delay": max_delay}})
    mock_score = mock_regression_score if pipeline_class == TimeSeriesBaselineRegressionPipeline else mock_classification_score
    if only_use_y:
        clf.fit(None, y)
        clf.score(X=None, y=y, objectives=['MCC Binary'])
    else:
        clf.fit(X, y)
        clf.score(X, y, objectives=['MCC Binary'])

    # Verify that NaNs are dropped before passed to objectives
    _, target, preds = mock_score.call_args[0]
    assert not target.isna().any()
    assert not preds.isna().any()

    # Target used for scoring matches expected dates
    pd.testing.assert_index_equal(target.index, expected_target.index)
    np.testing.assert_equal(target.values, expected_target.values)
