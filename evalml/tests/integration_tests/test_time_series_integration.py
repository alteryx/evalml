import pandas as pd
import pytest

from evalml.automl import AutoMLSearch
from evalml.problem_types import ProblemTypes


@pytest.mark.parametrize(
    "problem_type",
    [
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ],
)
def test_can_run_automl_for_time_series_with_categorical_and_boolean_features(
    problem_type,
):

    X = pd.DataFrame(
        {"features": range(101, 601), "date": pd.date_range("2010-10-01", periods=500)}
    )
    y = pd.Series(range(500))
    if problem_type == ProblemTypes.TIME_SERIES_BINARY:
        y = y % 2
    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        y = y % 3

    X.ww.init()
    X.ww["bool_feature"] = (
        pd.Series([True, False])
        .sample(n=X.shape[0], replace=True)
        .reset_index(drop=True)
    )
    X.ww["cat_feature"] = (
        pd.Series(["a", "b", "c"])
        .sample(n=X.shape[0], replace=True)
        .reset_index(drop=True)
    )

    automl = AutoMLSearch(
        X,
        y,
        problem_type=problem_type,
        problem_configuration={
            "max_delay": 5,
            "gap": 3,
            "forecast_horizon": 2,
            "date_index": "date",
        },
        optimize_thresholds=False,
    )
    automl.search()
    automl.best_pipeline.fit(X, y)
    X_valid = pd.DataFrame(
        {
            "date": pd.date_range(
                pd.Timestamp(X.date.iloc[-1]) + pd.Timedelta("4d"), periods=2
            )
        }
    )
    # Treat all features as not known-in-advanced
    automl.best_pipeline.predict(X_valid, X_train=X, y_train=y)
