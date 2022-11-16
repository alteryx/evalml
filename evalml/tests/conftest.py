import contextlib
import os
import sys
from datetime import datetime
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import py
import pytest
import woodwork as ww
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from skopt.space import Integer, Real
from woodwork import logical_types as ww_logical_types

from evalml.demos import load_weather
from evalml.model_family import ModelFamily
from evalml.objectives import BinaryClassificationObjective
from evalml.objectives.utils import get_core_objectives, get_non_core_objectives
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline,
)
from evalml.pipelines.components import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Estimator,
    LogisticRegressionClassifier,
)
from evalml.pipelines.components.ensemble.stacked_ensemble_base import (
    _nonstackable_model_families,
)
from evalml.pipelines.components.utils import _all_estimators
from evalml.preprocessing import load_data
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_binary,
    is_multiclass,
    is_regression,
    is_time_series,
)
from evalml.utils import infer_feature_types


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_offline: mark test to be skipped if offline (https://oss.alteryx.com cannot be reached)",
    )
    config.addinivalue_line(
        "markers",
        "skip_during_conda: mark test to be skipped if running during conda build",
    )


@pytest.fixture(scope="session")
def go():
    from plotly import graph_objects as go

    return go


@pytest.fixture(scope="session")
def im():
    from imblearn import over_sampling as im

    return im


@pytest.fixture(scope="session")
def lgbm():
    import lightgbm as lgbm

    return lgbm


@pytest.fixture(scope="session")
def vw():
    from vowpalwabbit import sklearn_vw as vw

    return vw


@pytest.fixture(scope="session")
def graphviz():
    import graphviz

    return graphviz


@pytest.fixture
def get_test_data_with_or_without_primary_key():
    def _get_test_data_with_primary_key(input_type, has_primary_key):
        X = None
        if input_type == "integer":
            X_dict = {
                "col_1_id": [0, 1, 2, 3],
                "col_2": [2, 3, 4, 5],
                "col_3_id": [1, 1, 2, 3],
                "col_5": [0, 0, 1, 2],
            }
            if not has_primary_key:
                X_dict["col_1_id"] = [1, 1, 2, 3]
            X = pd.DataFrame.from_dict(X_dict)

        elif input_type == "integer_nullable":
            X_dict = {
                "col_1_id": pd.Series([0, 1, 2, 3], dtype="Int64"),
                "col_2": pd.Series([2, 3, 4, 5], dtype="Int64"),
                "col_3_id": pd.Series([1, 1, 2, 3], dtype="Int64"),
                "col_5": pd.Series([0, 0, 1, 2], dtype="Int64"),
            }
            if not has_primary_key:
                X_dict["col_1_id"] = pd.Series([1, 1, 2, 3], dtype="Int64")
            X = pd.DataFrame.from_dict(X_dict)

        elif input_type == "double":
            X_dict = {
                "col_1_id": [0.0, 1.0, 2.0, 3.0],
                "col_2": [2, 3, 4, 5],
                "col_3_id": [1, 1, 2, 3],
                "col_5": [0, 0, 1, 2],
            }
            if not has_primary_key:
                X_dict["col_1_id"] = [1.0, 1.0, 2.0, 3.0]
            X = pd.DataFrame.from_dict(X_dict)

        elif input_type == "string":
            X_dict = {
                "col_1_id": ["a", "b", "c", "d"],
                "col_2": ["w", "x", "y", "z"],
                "col_3_id": [
                    "123456789012345",
                    "234567890123456",
                    "3456789012345678",
                    "45678901234567",
                ],
                "col_5": ["0", "0", "1", "2"],
            }
            if not has_primary_key:
                X_dict["col_1_id"] = ["b", "b", "c", "d"]
            X = pd.DataFrame.from_dict(X_dict)
            X.ww.init(
                logical_types={
                    "col_1_id": "categorical",
                    "col_2": "categorical",
                    "col_5": "categorical",
                },
            )
        return X

    return _get_test_data_with_primary_key


@pytest.fixture
def get_test_data_from_configuration():
    def _get_test_data_from_configuration(
        input_type,
        problem_type,
        column_names=None,
        nullable_target=False,
        scale=2,
    ):
        if is_time_series(problem_type) and "dates" not in column_names:
            column_names.append("dates")
        X_all = pd.DataFrame(
            {
                "all_null": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
                * scale,
                "int_null": [0, 1, 2, np.nan, 4, np.nan, 6, 7, 8, 9] * scale,
                "age_null": [0, 1, 2, np.nan, 4, np.nan, 6, 7, 8, 9] * scale,
                "bool_null": [
                    True,
                    None,
                    False,
                    True,
                    False,
                    None,
                    True,
                    True,
                    False,
                    True,
                ]
                * scale,
                "numerical": range(10 * scale),
                "categorical": ["a", "b", "a", "b", "b", "a", "b", "a", "a", "b"]
                * scale,
                "dates": pd.date_range("2000-02-03", periods=10 * scale, freq="W"),
                "text": [
                    "this is a string",
                    "this is another string",
                    "this is just another string",
                    "evalml should handle string input",
                    "cats are gr8",
                    "hello world",
                    "evalml is gr8",
                    "more strings",
                    "here we go",
                    "wheeeee!!!",
                ]
                * scale,
                "email": [
                    "abalone_0@gmail.com",
                    "AbaloneRings@yahoo.com",
                    "abalone_2@abalone.com",
                    "titanic_data@hotmail.com",
                    "fooEMAIL@email.org",
                    "evalml@evalml.org",
                    "evalml@alteryx.org",
                    "woodwork@alteryx.org",
                    "featuretools@alteryx.org",
                    "compose@alteryx.org",
                ]
                * scale,
                "url": [
                    "https://evalml.alteryx.com/en/stable/",
                    "https://woodwork.alteryx.com/en/stable/guides/statistical_insights.html",
                    "https://twitter.com/AlteryxOSS",
                    "https://www.twitter.com/AlteryxOSS",
                    "https://www.evalml.alteryx.com/en/stable/demos/text_input.html",
                    "https://github.com/alteryx/evalml",
                    "https://github.com/alteryx/featuretools",
                    "https://github.com/alteryx/woodwork",
                    "https://github.com/alteryx/compose",
                    "https://woodwork.alteryx.com/en/stable/",
                ]
                * scale,
                "ip": [
                    "0.0.0.0",
                    "1.1.1.101",
                    "1.1.101.1",
                    "1.101.1.1",
                    "101.1.1.1",
                    "192.168.1.1",
                    "255.255.255.255",
                    "2.1.1.101",
                    "2.1.101.1",
                    "2.101.1.1",
                ]
                * scale,
            },
        )
        y = pd.Series([0, 0, 1, 0, 0, 1, 1, 0, 1, 0] * scale)
        if problem_type == ProblemTypes.MULTICLASS:
            y = pd.Series([0, 2, 1, 2, 0, 2, 1, 2, 1, 0] * scale)
        elif is_regression(problem_type):
            y = pd.Series([1, 2, 3, 3, 3, 4, 5, 5, 6, 6] * scale)
        if nullable_target:
            y.iloc[2] = None
            if input_type == "ww":
                y = ww.init_series(y, logical_type="integer_nullable")
        X = X_all[column_names]

        if input_type == "np":
            X = X.to_numpy()
            y = y.to_numpy()

        if input_type == "ww":
            logical_types = {}
            if "text" in column_names:
                logical_types.update({"text": "NaturalLanguage"})
            if "categorical" in column_names:
                logical_types.update({"categorical": "Categorical"})
            if "url" in column_names:
                logical_types.update({"url": "URL"})
            if "email" in column_names:
                logical_types.update({"email": "EmailAddress"})
            if "int_null" in column_names:
                logical_types.update({"int_null": "integer_nullable"})
            if "age_null" in column_names:
                logical_types.update({"age_null": "age_nullable"})
            if "bool_null" in column_names:
                logical_types.update({"bool_null": "boolean_nullable"})

            X.ww.init(logical_types=logical_types)

            y = ww.init_series(y)

        return X, y

    return _get_test_data_from_configuration


@pytest.fixture
def ts_data():
    def _get_X_y(
        train_features_index_dt=True,
        train_target_index_dt=True,
        train_none=False,
        datetime_feature=True,
        no_features=False,
        test_features_index_dt=True,
        freq="D",
        problem_type="time series regression",
    ):
        X = pd.DataFrame(index=[i + 1 for i in range(50)])
        dates = pd.date_range("1/1/21", periods=50, freq=freq)
        feature = pd.Series([1, 4, 2] * 10 + [3, 1] * 10, index=X.index)
        y = pd.Series([1, 2, 3, 4, 5, 6, 5, 4, 3, 2] * 5)

        X_train = X.iloc[:40]
        X_test = X.iloc[40:]
        y_train = y.iloc[:40]
        logical_types = {}

        if train_features_index_dt:
            X_train.index = dates[:40]
        if train_target_index_dt:
            y_train.index = dates[:40]
        if test_features_index_dt:
            X_test.index = dates[40:]
        if not no_features:
            X_train["feature"] = pd.Series(feature[:40].values, index=X_train.index)
            X_test["feature"] = pd.Series(feature[40:].values, index=X_test.index)
            logical_types["feature"] = "integer"
        if datetime_feature:
            X_train["date"] = pd.Series(dates[:40].values, index=X_train.index)
            X_test["date"] = pd.Series(dates[40:].values, index=X_test.index)
            logical_types["date"] = "datetime"
        if train_none:
            X_train = None
        if is_binary(problem_type):
            y_train = y_train % 2
        if is_multiclass(problem_type):
            y_train = y_train % 3

        if X_train is not None:
            X_train.ww.init(logical_types=logical_types)
        X_test.ww.init(logical_types=logical_types)
        y_train.ww.init(logical_type="integer")

        return X_train, X_test, y_train

    return _get_X_y


def create_mock_pipeline(
    estimator,
    problem_type,
    parameters=None,
    add_label_encoder=False,
):
    pipeline_parameters = (
        {estimator.name: {"n_jobs": 1}}
        if (
            estimator.model_family
            not in [
                ModelFamily.SVM,
                ModelFamily.DECISION_TREE,
                ModelFamily.VOWPAL_WABBIT,
                ModelFamily.PROPHET,
            ]
            and "Elastic Net" not in estimator.name
        )
        else {}
    )

    if parameters is not None:
        pipeline_parameters.update(parameters)

    custom_name = (
        f"Pipeline with {estimator.name}"
        if add_label_encoder is False
        else f"Pipeline with {estimator.name} w/ Label Encoder"
    )
    component_graph = (
        [estimator]
        if add_label_encoder is False
        else {
            "Label Encoder": ["Label Encoder", "X", "y"],
            estimator.name: [
                estimator,
                "Label Encoder.x",
                "Label Encoder.y",
            ],
        }
    )

    if problem_type == ProblemTypes.BINARY:
        return BinaryClassificationPipeline(
            component_graph,
            parameters=pipeline_parameters,
            custom_name=custom_name,
        )
    elif problem_type == ProblemTypes.MULTICLASS:
        return MulticlassClassificationPipeline(
            component_graph,
            parameters=pipeline_parameters,
            custom_name=custom_name,
        )
    elif problem_type == ProblemTypes.REGRESSION:
        return RegressionPipeline(
            component_graph,
            parameters=pipeline_parameters,
            custom_name=custom_name,
        )
    elif problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        return TimeSeriesRegressionPipeline(
            component_graph,
            parameters=pipeline_parameters,
            custom_name=custom_name,
        )
    elif problem_type == ProblemTypes.TIME_SERIES_BINARY:
        return TimeSeriesBinaryClassificationPipeline(
            component_graph,
            parameters=pipeline_parameters,
            custom_name=custom_name,
        )
    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        return TimeSeriesMulticlassClassificationPipeline(
            component_graph,
            parameters=pipeline_parameters,
            custom_name=custom_name,
        )


@pytest.fixture
def all_pipeline_classes():
    ts_parameters = {
        "pipeline": {
            "time_index": "date",
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 3,
        },
    }

    all_possible_pipeline_classes = []
    for estimator in _all_estimators():
        for problem_type in estimator.supported_problem_types:

            all_possible_pipeline_classes.append(
                create_mock_pipeline(
                    estimator,
                    problem_type,
                    parameters=ts_parameters if is_time_series(problem_type) else None,
                    add_label_encoder=False,
                ),
            )
            all_possible_pipeline_classes.append(
                create_mock_pipeline(
                    estimator,
                    problem_type,
                    parameters=ts_parameters if is_time_series(problem_type) else None,
                    add_label_encoder=True,
                ),
            )
    return all_possible_pipeline_classes


@pytest.fixture
def all_binary_pipeline_classes(all_pipeline_classes):
    return [
        pipeline
        for pipeline in all_pipeline_classes
        if isinstance(pipeline, BinaryClassificationPipeline)
        and "label encoder" not in pipeline.custom_name
    ]


@pytest.fixture
def all_binary_pipeline_classes_with_encoder(all_pipeline_classes):
    return [
        pipeline
        for pipeline in all_pipeline_classes
        if isinstance(pipeline, BinaryClassificationPipeline)
        and "label encoder" in pipeline.custom_name
    ]


@pytest.fixture
def all_multiclass_pipeline_classes(all_pipeline_classes):
    return [
        pipeline
        for pipeline in all_pipeline_classes
        if isinstance(pipeline, MulticlassClassificationPipeline)
        and "label encoder" not in pipeline.custom_name
    ]


@pytest.fixture
def all_multiclass_pipeline_classes_with_encoder(all_pipeline_classes):
    return [
        pipeline
        for pipeline in all_pipeline_classes
        if isinstance(pipeline, MulticlassClassificationPipeline)
        and "label encoder" in pipeline.custom_name
    ]


def pytest_addoption(parser):
    parser.addoption(
        "--is-using-conda",
        action="store_true",
        default=False,
        help="If true, tests will assume that they are being run as part of"
        "the build_conda_pkg workflow with the feedstock.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--is-using-conda"):
        skip_conda = pytest.mark.skip(reason="Test does not run during conda")
        for item in items:
            if "skip_during_conda" in item.keywords:
                item.add_marker(skip_conda)


@pytest.fixture
def is_using_conda(pytestconfig):
    return pytestconfig.getoption("--is-using-conda")


@pytest.fixture
def is_using_windows(pytestconfig):
    return sys.platform in ["win32", "cygwin"]


@pytest.fixture
def assert_allowed_pipelines_equal_helper():
    def assert_allowed_pipelines_equal_helper(
        actual_allowed_pipelines,
        expected_allowed_pipelines,
    ):
        actual_allowed_pipelines.sort(key=lambda p: p.name)
        expected_allowed_pipelines.sort(key=lambda p: p.name)
        for actual, expected in zip(
            actual_allowed_pipelines,
            expected_allowed_pipelines,
        ):
            for pipeline_subclass in [
                BinaryClassificationPipeline,
                MulticlassClassificationPipeline,
                RegressionPipeline,
            ]:
                if isinstance(expected, pipeline_subclass):
                    assert isinstance(expected, pipeline_subclass)
                    break
            assert actual.parameters == expected.parameters
            assert actual.name == expected.name
            assert actual.problem_type == expected.problem_type
            assert actual.component_graph == expected.component_graph

    return assert_allowed_pipelines_equal_helper


@pytest.fixture
def missing_middle():
    dates_1 = pd.date_range("2005-01-01 00:00:00", periods=20, freq="1H")
    dates_2 = pd.date_range("2005-01-01 21:00:00", periods=30, freq="1H")
    dates = dates_1.append(dates_2)
    return dates


@pytest.fixture
def missing_beginning():
    dates_1 = pd.DatetimeIndex(["2006-08-01"])
    dates_2 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates = dates_1.append(dates_2)
    return dates


@pytest.fixture
def missing_end():
    dates_1 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates_2 = pd.DatetimeIndex(["2009-05-01"])
    dates = dates_1.append(dates_2)
    return dates


@pytest.fixture
def missing_scattered():
    dates_1 = pd.date_range("2014-03-01", periods=27, freq="5D")
    dates_2 = pd.date_range("2014-07-19", periods=9, freq="5D")
    dates_3 = pd.date_range("2014-09-07", periods=15, freq="5D")
    dates_4 = pd.date_range("2014-12-01", periods=21, freq="5D")
    dates = dates_1.append(dates_2).append(dates_3).append(dates_4)
    return dates


@pytest.fixture
def missing_continuous():
    dates_1 = pd.date_range("2014-03-01", periods=22, freq="5D")
    dates_2 = pd.date_range("2014-06-29", periods=13, freq="5D")
    dates_3 = pd.date_range("2014-09-17", periods=31, freq="5D")
    dates = dates_1.append(dates_2).append(dates_3)
    return dates


@pytest.fixture
def uneven_middle():
    dates_1 = pd.date_range("2005-01-01 00:00:00", periods=20, freq="10H")
    dates_2 = pd.DatetimeIndex(["2005-01-09 12:00:00"])
    dates_3 = pd.DatetimeIndex(["2005-01-09 20:00:00"])
    dates_4 = pd.date_range("2005-01-10 04:00:00", periods=29, freq="10H")
    dates = dates_1.append(dates_2).append(dates_3).append(dates_4)
    return dates


@pytest.fixture
def uneven_beginning():
    dates_1 = pd.DatetimeIndex(["2006-07-23", "2006-08-21"])
    dates_2 = pd.date_range("2006-10-31", periods=30, freq="2M")
    dates = dates_1.append(dates_2)
    return dates


@pytest.fixture
def uneven_end():
    dates_1 = pd.date_range("2006-10-01", periods=30, freq="1MS")
    dates_2 = pd.DatetimeIndex(["2009-03-26"])
    dates = dates_1.append(dates_2)
    return dates


@pytest.fixture
def uneven_scattered():
    dates_1 = pd.date_range("2014-03-01", periods=17, freq="5D")
    dates_2 = pd.DatetimeIndex(["2014-05-23"])
    dates_3 = pd.date_range("2014-05-30", periods=8, freq="5D")
    dates_4 = pd.DatetimeIndex(["2014-07-11"])
    dates_5 = pd.date_range("2014-07-14", periods=13, freq="5D")
    dates_6 = pd.DatetimeIndex(["2014-09-13", "2014-09-18"])
    dates_7 = pd.date_range("2014-09-27", periods=159, freq="5D")
    dates = (
        dates_1.append(dates_2)
        .append(dates_3)
        .append(dates_4)
        .append(dates_5)
        .append(dates_6)
        .append(dates_7)
    )
    return dates


@pytest.fixture
def uneven_work_week():
    dates_1 = pd.date_range("2015-01-12", periods=30, freq="B")
    dates_2 = pd.date_range("2015-02-23", periods=3, freq="D")
    dates_3 = pd.DatetimeIndex(["2015-02-27"])
    dates_4 = pd.date_range("2015-03-02", periods=40, freq="B")

    dates = dates_1.append(dates_2).append(dates_3).append(dates_4)
    return dates


@pytest.fixture
def uneven_continuous():
    dates_1 = pd.date_range("2014-01-01", periods=5, freq="2D")
    dates_2 = pd.DatetimeIndex(["2014-01-10", "2014-01-12", "2014-01-14"])
    dates_3 = pd.date_range("2014-01-15", periods=130, freq="2D")
    dates = dates_1.append(dates_2).append(dates_3)
    return dates


@pytest.fixture
def duplicate_beginning():
    dates_1 = pd.DatetimeIndex(["2006-09-01"])
    dates_2 = pd.date_range("2006-09-01", periods=30, freq="1MS")
    dates = dates_1.append(dates_2)
    return dates


@pytest.fixture
def duplicate_middle():
    dates_1 = pd.date_range("2005-01-01", periods=10, freq="2MS")
    dates_2 = pd.DatetimeIndex(["2006-07-01"])
    dates_3 = pd.date_range("2006-09-01", periods=50, freq="2MS")
    dates = dates_1.append(dates_2).append(dates_3)
    return dates


@pytest.fixture
def duplicate_end():
    dates_1 = pd.date_range("2001-01-07", periods=30, freq="3D")
    dates_2 = pd.DatetimeIndex(["2001-04-10", "2001-04-10"])
    dates = dates_1.append(dates_2)
    return dates


@pytest.fixture
def duplicate_scattered():
    dates_1 = pd.DatetimeIndex(["2001-01-01", "2001-01-01"])
    dates_2 = pd.date_range("2001-01-01", periods=20, freq="1AS")
    dates_3 = pd.DatetimeIndex(["2020-01-01"])
    dates_4 = pd.date_range("2022-01-01", periods=5, freq="1AS")
    dates_5 = pd.DatetimeIndex(["2026-01-01"])
    dates = dates_1.append(dates_2).append(dates_3).append(dates_4).append(dates_5)
    return dates


@pytest.fixture
def duplicate_continuous():
    dates_1 = pd.date_range("2005-01-01", periods=10, freq="2MS")
    dates_2 = pd.DatetimeIndex(["2006-07-01", "2006-07-01", "2006-07-01"])
    dates_3 = pd.date_range("2006-09-01", periods=50, freq="2MS")
    dates = dates_1.append(dates_2).append(dates_3)
    return dates


@pytest.fixture
def combination_of_faulty_datetime():
    dates_0 = pd.DatetimeIndex(["1/1/21", "1/3/21", "1/3/21", "1/3/21"])
    dates_1 = pd.date_range("1/5/21", periods=8, freq="2D")
    dates_2 = pd.DatetimeIndex(["1/19/21"])
    dates_3 = pd.date_range("1/27/21", periods=7, freq="2D")
    dates_4 = pd.date_range("2/12/21", periods=13, freq="2D")
    dates_5 = pd.DatetimeIndex(
        [
            "3/09/21",
            "3/10/21",
            "3/12/21",
            "3/14/21",
            None,
            "3/18/21",
            "3/20/21",
            "3/22/21",
            "3/22/21",
        ],
    )
    dates_6 = pd.date_range("3/22/21", periods=10, freq="2D")
    dates_7 = pd.DatetimeIndex(["4/12/21", "4/14/21"])
    dates_8 = pd.date_range("4/15/21", periods=5, freq="2D")
    dates_9 = pd.DatetimeIndex(["4/24/21"])
    dates_10 = pd.date_range("4/25/21", periods=3, freq="2D")
    dates_11 = pd.DatetimeIndex([None, None, None])
    dates_12 = pd.date_range("5/5/21", periods=300, freq="2D")

    dates = pd.DatetimeIndex([])
    for dates_ in [
        dates_0,
        dates_1,
        dates_2,
        dates_3,
        dates_4,
        dates_5,
        dates_6,
        dates_7,
        dates_8,
        dates_9,
        dates_10,
        dates_11,
        dates_12,
    ]:
        dates = dates.append(dates_)
    return dates


@pytest.fixture
def X_y_binary():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        random_state=0,
    )
    X = pd.DataFrame(X)
    X.ww.init(logical_types={col: "double" for col in X.columns})
    y = ww.init_series(pd.Series(y), logical_type="integer")
    return X, y


@pytest.fixture(scope="session")
def X_y_binary_cls():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=20,
        n_informative=2,
        n_redundant=2,
        random_state=0,
    )
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def X_y_regression():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=20,
        n_informative=3,
        random_state=0,
    )
    X = pd.DataFrame(X)
    X.ww.init(logical_types={col: "double" for col in X.columns})
    y = ww.init_series(pd.Series(y), logical_type="double")
    return X, y


@pytest.fixture
def X_y_multi():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=20,
        n_classes=3,
        n_informative=3,
        n_redundant=2,
        random_state=0,
    )
    X = pd.DataFrame(X)
    X.ww.init(logical_types={col: "double" for col in X.columns})
    y = ww.init_series(pd.Series(y), logical_type="integer")
    return X, y


@pytest.fixture
def X_y_categorical_regression():
    data_path = os.path.join(os.path.dirname(__file__), "data/tips.csv")
    flights = pd.read_csv(data_path)

    y = flights["tip"]
    X = flights.drop("tip", axis=1)

    # add categorical dtype
    X["smoker"] = X["smoker"].astype("category")

    X.ww.init(
        logical_types={
            "total_bill": "double",
            "sex": "categorical",
            "smoker": "categorical",
            "day": "categorical",
            "time": "categorical",
            "size": "integer",
        },
    )
    y.ww.init(logical_type="double")
    return X, y


@pytest.fixture
def X_y_categorical_classification():
    data_path = os.path.join(os.path.dirname(__file__), "data/titanic.csv")
    titanic = pd.read_csv(data_path)

    y = titanic["Survived"]
    X = titanic.drop(["Survived", "Name"], axis=1)
    X.ww.init(
        logical_types={
            "PassengerId": "integer",
            "Pclass": "integer",
            "Sex": "categorical",
            "Age": "double",
            "SibSp": "integer",
            "Parch": "integer",
            "Ticket": "categorical",
            "Fare": "double",
            "Cabin": "categorical",
            "Embarked": "categorical",
        },
    )
    y.ww.init(logical_type="integer")
    return X, y


@pytest.fixture
def X_y_based_on_pipeline_or_problem_type(X_y_binary, X_y_multi, X_y_regression):
    def _X_y_based_on_pipeline_or_problem_type(pipeline_or_type):
        problem_types = {
            ProblemTypes.BINARY: "binary",
            ProblemTypes.MULTICLASS: "multiclass",
            ProblemTypes.REGRESSION: "regression",
            ProblemTypes.TIME_SERIES_BINARY: "binary",
            ProblemTypes.TIME_SERIES_MULTICLASS: "multiclass",
            ProblemTypes.TIME_SERIES_REGRESSION: "regression",
        }
        pipeline_classes = {
            BinaryClassificationPipeline: "binary",
            MulticlassClassificationPipeline: "multiclass",
            RegressionPipeline: "regression",
        }

        if pipeline_or_type in problem_types:
            problem_type = problem_types[pipeline_or_type]
        elif pipeline_or_type in pipeline_classes:
            problem_type = pipeline_classes[pipeline_or_type]

        if problem_type == "binary":
            X, y = X_y_binary
        elif problem_type == "multiclass":
            X, y = X_y_multi
        else:
            X, y = X_y_regression
        return X, y

    return _X_y_based_on_pipeline_or_problem_type


@pytest.fixture()
def text_df():
    df = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
            ],
            "col_2": [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living",
                "Red, the blood of angry men - black, the dark of ages past",
            ],
        },
    )
    df.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    yield df


@pytest.fixture
def ts_data_long():
    X, y = pd.DataFrame(
        {
            "features": range(101, 193),
            "date": pd.date_range("2020-10-01", "2020-12-31"),
        },
    ), pd.Series(range(1, 93))
    y.index = pd.date_range("2020-10-01", "2020-12-31")
    X.index = pd.date_range("2020-10-01", "2020-12-31")
    return X, None, y


@pytest.fixture
def ts_data_quadratic_trend(ts_data):
    X, _, y = ts_data()
    y = y**2
    return X, y


@pytest.fixture
def ts_data_cubic_trend(ts_data):
    X, _, y = ts_data()
    y = y**3
    return X, y


@pytest.fixture
def ts_data_seasonal_train():
    sine_ = np.linspace(-np.pi * 5, np.pi * 5, 25)
    X, y = pd.DataFrame({"features": range(25)}), pd.Series(sine_)
    y.index = pd.date_range(start="1/1/2018", periods=25)
    X.index = pd.date_range(start="1/1/2018", periods=25)
    return X, y


@pytest.fixture
def ts_data_seasonal_test():
    sine_ = np.linspace(-np.pi * 5, np.pi * 5, 25)
    X, y = pd.DataFrame({"features": range(25)}), pd.Series(sine_)
    y.index = pd.date_range(start="1/26/2018", periods=25)
    X.index = pd.date_range(start="1/26/2018", periods=25)
    return X, y


@pytest.fixture
def dummy_pipeline_hyperparameters():
    return {
        "Mock Classifier": {
            "param a": Integer(0, 10),
            "param b": Real(0, 10),
            "param c": ["option a", "option b", "option c"],
            "param d": ["option a", "option b", 100, np.inf],
        },
    }


@pytest.fixture
def dummy_pipeline_hyperparameters_unicode():
    return {
        "Mock Classifier": {
            "param a": Integer(0, 10),
            "param b": Real(0, 10),
            "param c": ["option a 💩", "option b 💩", "option c 💩"],
            "param d": ["option a", "option b", 100, np.inf],
        },
    }


@pytest.fixture
def dummy_pipeline_hyperparameters_small():
    return {
        "Mock Classifier": {
            "param a": ["most_frequent", "median", "mean"],
            "param b": ["a", "b", "c"],
        },
    }


@pytest.fixture
def dummy_classifier_estimator_class():
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.NONE
        supported_problem_types = [
            ProblemTypes.BINARY,
            ProblemTypes.MULTICLASS,
            ProblemTypes.TIME_SERIES_MULTICLASS,
            ProblemTypes.TIME_SERIES_BINARY,
        ]
        hyperparameter_ranges = {"a": Integer(0, 10), "b": Real(0, 10)}

        def __init__(self, a=1, b=0, random_seed=0):
            super().__init__(
                parameters={"a": a, "b": b},
                component_obj=None,
                random_seed=random_seed,
            )

        def fit(self, X, y):
            return self

    return MockEstimator


@pytest.fixture
def example_graph():
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Classifier", "OneHot_ElasticNet.x", "y"],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    return component_graph


@pytest.fixture
def example_graph_with_transformer_last_component():
    component_graph = {
        "Label Encoder": ["Label Encoder", "X", "y"],
        "Imputer": ["Imputer", "X", "Label Encoder.y"],
        "OneHotEncoder": ["One Hot Encoder", "Imputer.x", "Label Encoder.y"],
    }
    return component_graph


@pytest.fixture
def example_pass_target_graph():
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "Target Imputer": ["Target Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "Target Imputer.y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Classifier", "OneHot_ElasticNet.x", "y"],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    return component_graph


@pytest.fixture
def example_regression_graph():
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Regressor", "OneHot.x", "y"],
        "Elastic Net": ["Elastic Net Regressor", "OneHot.x", "y"],
        "Linear Regressor": [
            "Linear Regressor",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    return component_graph


@pytest.fixture
def dummy_binary_pipeline(dummy_classifier_estimator_class):
    return BinaryClassificationPipeline(
        component_graph=[dummy_classifier_estimator_class],
        custom_name="Mock Binary Classification Pipeline",
    )


@pytest.fixture
def dummy_multiclass_pipeline(dummy_classifier_estimator_class):
    return MulticlassClassificationPipeline(
        component_graph=[dummy_classifier_estimator_class],
        custom_name="Mock Multiclass Classification Pipeline",
    )


@pytest.fixture
def dummy_regressor_estimator_class():
    class MockRegressor(Estimator):
        name = "Mock Regressor"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.REGRESSION]
        hyperparameter_ranges = {"a": Integer(0, 10), "b": Real(0, 10)}

        def __init__(self, a=1, b=0, random_seed=0):
            super().__init__(
                parameters={"a": a, "b": b},
                component_obj=None,
                random_seed=random_seed,
            )

        def fit(self, X, y):
            return self

    return MockRegressor


@pytest.fixture
def dummy_regression_pipeline(dummy_regressor_estimator_class):
    return RegressionPipeline(
        component_graph=[dummy_regressor_estimator_class],
        custom_name="Mock Regression Pipeline",
    )


@pytest.fixture
def dummy_time_series_regressor_estimator_class():
    class MockTimeSeriesRegressor(Estimator):
        name = "Mock Time Series Regressor"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
        hyperparameter_ranges = {"a": Integer(0, 10), "b": Real(0, 10)}

        def __init__(self, a=1, b=0, random_seed=0):
            super().__init__(
                parameters={"a": a, "b": b},
                component_obj=None,
                random_seed=random_seed,
            )

    return MockTimeSeriesRegressor


@pytest.fixture
def dummy_time_series_regression_pipeline_class(
    dummy_time_series_regressor_estimator_class,
):
    MockTimeSeriesRegressor = dummy_time_series_regressor_estimator_class

    class MockTimeSeriesRegressionPipeline(TimeSeriesRegressionPipeline):
        component_graph = [MockTimeSeriesRegressor]
        custom_name = None

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    return MockTimeSeriesRegressionPipeline


@pytest.fixture
def dummy_ts_binary_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockBinaryClassificationPipeline(TimeSeriesBinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator]

        def __init__(
            self,
            parameters,
            custom_name=None,
            component_graph=None,
            random_seed=0,
        ):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                random_seed=random_seed,
            )

    return MockBinaryClassificationPipeline


@pytest.fixture
def dummy_ts_binary_tree_classifier_pipeline_class():
    dec_tree_classifier = DecisionTreeClassifier

    class MockBinaryClassificationPipeline(TimeSeriesBinaryClassificationPipeline):
        estimator = dec_tree_classifier
        component_graph = ["DateTime Featurizer", estimator]

        def __init__(
            self,
            parameters,
            custom_name=None,
            component_graph=None,
            random_seed=0,
        ):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                random_seed=random_seed,
            )

    return MockBinaryClassificationPipeline


@pytest.fixture
def dummy_ts_multi_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockMultiClassificationClassificationPipeline(
        TimeSeriesMulticlassClassificationPipeline,
    ):
        estimator = MockEstimator
        component_graph = [MockEstimator]

        def __init__(
            self,
            parameters,
            custom_name=None,
            component_graph=None,
            random_seed=0,
        ):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                random_seed=random_seed,
            )

    return MockMultiClassificationClassificationPipeline


@pytest.fixture
def logistic_regression_component_graph():
    component_graph = {
        "Label Encoder": ["Label Encoder", "X", "y"],
        "Imputer": ["Imputer", "X", "Label Encoder.y"],
        "One Hot Encoder": ["One Hot Encoder", "Imputer.x", "Label Encoder.y"],
        "Standard Scaler": [
            "Standard Scaler",
            "One Hot Encoder.x",
            "Label Encoder.y",
        ],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Standard Scaler.x",
            "Label Encoder.y",
        ],
    }
    return component_graph


@pytest.fixture
def logistic_regression_multiclass_pipeline(logistic_regression_component_graph):
    return MulticlassClassificationPipeline(
        component_graph=logistic_regression_component_graph,
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
        custom_name="Logistic Regression Multiclass Pipeline",
    )


@pytest.fixture
def logistic_regression_binary_pipeline(logistic_regression_component_graph):
    return BinaryClassificationPipeline(
        component_graph=logistic_regression_component_graph,
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
        custom_name="Logistic Regression Binary Pipeline",
    )


@pytest.fixture
def linear_regression_pipeline():
    return RegressionPipeline(
        component_graph=[
            "One Hot Encoder",
            "Imputer",
            "Standard Scaler",
            "Linear Regressor",
        ],
        parameters={"Linear Regressor": {"n_jobs": 1}},
        custom_name="Linear Regression Pipeline",
    )


@pytest.fixture
def time_series_regression_pipeline_class():
    class TSRegressionPipeline(TimeSeriesRegressionPipeline):
        """Random Forest Regression Pipeline for time series regression problems."""

        component_graph = {
            "Time Series Featurizer": [
                "Time Series Featurizer",
                "X",
                "y",
            ],
            "DateTime Featurizer": [
                "DateTime Featurizer",
                "Time Series Featurizer.x",
                "y",
            ],
            "Drop NaN Rows Transformer": [
                "Drop NaN Rows Transformer",
                "DateTime Featurizer.x",
                "y",
            ],
            "Random Forest Regressor": [
                "Random Forest Regressor",
                "Drop NaN Rows Transformer.x",
                "Drop NaN Rows Transformer.y",
            ],
        }

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                random_seed=random_seed,
            )

    return TSRegressionPipeline


@pytest.fixture
def time_series_classification_component_graph():
    component_graph = {
        "Label Encoder": ["Label Encoder", "X", "y"],
        "Time Series Featurizer": [
            "Time Series Featurizer",
            "Label Encoder.x",
            "Label Encoder.y",
        ],
        "DateTime Featurizer": [
            "DateTime Featurizer",
            "Time Series Featurizer.x",
            "Label Encoder.y",
        ],
        "Drop NaN Rows Transformer": [
            "Drop NaN Rows Transformer",
            "DateTime Featurizer.x",
            "Label Encoder.y",
        ],
        "Logistic Regression Classifier": [
            "Logistic Regression Classifier",
            "Drop NaN Rows Transformer.x",
            "Drop NaN Rows Transformer.y",
        ],
    }
    return component_graph


@pytest.fixture
def time_series_binary_classification_pipeline_class(
    time_series_classification_component_graph,
):
    class TSBinaryPipeline(TimeSeriesBinaryClassificationPipeline):
        """Logistic Regression Pipeline for time series binary classification problems."""

        component_graph = time_series_classification_component_graph

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                random_seed=random_seed,
            )

    return TSBinaryPipeline


@pytest.fixture
def time_series_multiclass_classification_pipeline_class(
    time_series_classification_component_graph,
):
    class TSMultiPipeline(TimeSeriesMulticlassClassificationPipeline):
        """Logistic Regression Pipeline for time series multiclass classification problems."""

        component_graph = time_series_classification_component_graph

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                random_seed=random_seed,
            )

    return TSMultiPipeline


@pytest.fixture
def fitted_decision_tree_classification_pipeline(X_y_categorical_classification):
    pipeline = BinaryClassificationPipeline(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "OneHot": ["One Hot Encoder", "Imputer.x", "y"],
            "Standard Scaler": ["Standard Scaler", "OneHot.x", "y"],
            "Decision Tree Classifier": [
                "Elastic Net Classifier",
                "Standard Scaler.x",
                "y",
            ],
        },
    )
    X, y = X_y_categorical_classification
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def nonlinear_binary_pipeline(example_graph):
    return BinaryClassificationPipeline(
        component_graph=example_graph,
        custom_name="Non Linear Binary Pipeline",
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
    )


@pytest.fixture
def nonlinear_binary_with_target_pipeline(example_pass_target_graph):
    return BinaryClassificationPipeline(
        component_graph=example_pass_target_graph,
        custom_name="Non Linear Binary With Target Pipeline",
    )


@pytest.fixture
def nonlinear_multiclass_pipeline(example_graph):
    return MulticlassClassificationPipeline(
        component_graph=example_graph,
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}},
    )


@pytest.fixture
def nonlinear_regression_pipeline(example_regression_graph):
    return RegressionPipeline(
        component_graph=example_regression_graph,
        parameters={"Linear Regressor": {"n_jobs": 1}},
    )


@pytest.fixture
def binary_test_objectives():
    return [
        o
        for o in get_core_objectives(ProblemTypes.BINARY)
        if o.name in {"Log Loss Binary", "F1", "AUC"}
    ]


@pytest.fixture
def multiclass_test_objectives():
    return [
        o
        for o in get_core_objectives(ProblemTypes.MULTICLASS)
        if o.name in {"Log Loss Multiclass", "AUC Micro", "F1 Micro"}
    ]


@pytest.fixture
def regression_test_objectives():
    return [
        o
        for o in get_core_objectives(ProblemTypes.REGRESSION)
        if o.name in {"R2", "Root Mean Squared Error", "MAE"}
    ]


@pytest.fixture
def time_series_core_objectives():
    return get_core_objectives(ProblemTypes.TIME_SERIES_REGRESSION)


@pytest.fixture
def time_series_non_core_objectives():
    non_core_time_series = [
        obj_()
        for obj_ in get_non_core_objectives()
        if ProblemTypes.TIME_SERIES_REGRESSION in obj_.problem_types
    ]
    return non_core_time_series


@pytest.fixture
def time_series_objectives(
    time_series_core_objectives,
    time_series_non_core_objectives,
):
    return time_series_core_objectives + time_series_non_core_objectives


@pytest.fixture
def stackable_classifiers():
    stackable_classifiers = []
    for estimator_class in _all_estimators():
        supported_problem_types = [
            handle_problem_types(pt) for pt in estimator_class.supported_problem_types
        ]
        if (
            set(supported_problem_types)
            == {
                ProblemTypes.BINARY,
                ProblemTypes.MULTICLASS,
                ProblemTypes.TIME_SERIES_BINARY,
                ProblemTypes.TIME_SERIES_MULTICLASS,
            }
            and estimator_class.model_family not in _nonstackable_model_families
            and estimator_class.model_family != ModelFamily.ENSEMBLE
        ):
            stackable_classifiers.append(estimator_class)
    return stackable_classifiers


@pytest.fixture
def stackable_regressors():
    stackable_regressors = []
    for estimator_class in _all_estimators():
        supported_problem_types = [
            handle_problem_types(pt) for pt in estimator_class.supported_problem_types
        ]
        if (
            set(supported_problem_types)
            == {ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION}
            and estimator_class.model_family not in _nonstackable_model_families
            and estimator_class.model_family != ModelFamily.ENSEMBLE
        ):
            stackable_regressors.append(estimator_class)
    return stackable_regressors


@pytest.fixture
def tree_estimators():
    est_classifier_class = DecisionTreeClassifier()
    est_regressor_class = DecisionTreeRegressor()
    return est_classifier_class, est_regressor_class


@pytest.fixture
def fitted_tree_estimators(tree_estimators, X_y_binary, X_y_regression):
    est_clf, est_reg = tree_estimators
    X_b, y_b = X_y_binary
    X_r, y_r = X_y_regression

    b_cols = [f"Testing_{col}" for col in X_b.columns]
    X_b = pd.DataFrame(X_b)
    X_b.columns = b_cols
    X_b.ww.init(logical_types={col: "double" for col in X_b.columns})

    r_cols = [f"Testing_{col}" for col in X_r.columns]
    X_r = pd.DataFrame(
        X_r,
    )
    X_r.columns = r_cols
    X_r.ww.init(logical_types={col: "double" for col in X_r.columns})

    est_clf.fit(X_b, y_b)
    est_reg.fit(X_r, y_r)
    return est_clf, est_reg


@pytest.fixture
def logit_estimator():
    est_class = LogisticRegressionClassifier()
    return est_class


@pytest.fixture
def helper_functions():
    class Helpers:
        @staticmethod
        def safe_init_component_with_njobs_1(component_class):
            try:
                component = component_class(n_jobs=1)
            except TypeError:
                component = component_class()
            return component

    return Helpers


@pytest.fixture
def make_data_type():
    """Helper function to convert numpy or pandas input to the appropriate type for tests."""

    def _make_data_type(data_type, data, nullable=False):
        if data_type == "li":
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
            data = data.tolist()
            return data
        if data_type != "np":
            if len(data.shape) == 1:
                data = pd.Series(data)
            else:
                data = pd.DataFrame(data)
        if data_type == "ww":
            if len(data.shape) == 1:
                data = ww.init_series(data)
                if nullable and isinstance(
                    data.ww.logical_type,
                    ww_logical_types.Integer,
                ):
                    data = ww.init_series(
                        data,
                        logical_type=ww_logical_types.IntegerNullable,
                    )
                elif nullable and isinstance(
                    data.ww.logical_type,
                    ww_logical_types.Boolean,
                ):
                    data = ww.init_series(
                        data,
                        logical_type=ww_logical_types.BooleanNullable,
                    )
            else:
                data.ww.init()
        return data

    return _make_data_type


def load_fraud_local(n_rows=None):
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    fraud_data_path = os.path.join(data_folder_path, "fraud_transactions.csv.gz")
    X, y = load_data(
        path=fraud_data_path,
        index="id",
        target="fraud",
        compression="gzip",
        n_rows=n_rows,
    )
    return X, y


@pytest.fixture
def fraud_local():
    X, y = load_fraud_local()
    X.ww.set_types(logical_types={"provider": "Categorical", "region": "Categorical"})
    return X, y


@pytest.fixture
def fraud_100():
    X, y = load_fraud_local(n_rows=100)
    X.ww.set_types(
        logical_types={
            "provider": "Categorical",
            "region": "Categorical",
            "currency": "categorical",
            "expiration_date": "categorical",
        },
    )
    return X, y


@pytest.fixture
def breast_cancer_local():
    data = datasets.load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    y = y.map(lambda x: data["target_names"][x])
    X.ww.init()
    y = ww.init_series(y)
    return X, y


@pytest.fixture
def wine_local():
    data = datasets.load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    y = y.map(lambda x: data["target_names"][x])
    X.ww.init()
    y = ww.init_series(y)
    return X, y


@pytest.fixture
def diabetes_local():
    data = datasets.load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X.ww.init()
    y = ww.init_series(y)
    return X, y


@pytest.fixture
def churn_local():
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    churn_data_path = os.path.join(data_folder_path, "churn.csv")
    return load_data(
        path=churn_data_path,
        index="customerID",
        target="Churn",
    )


@pytest.fixture
def mock_imbalanced_data_X_y():
    """Helper function to return an imbalanced binary or multiclass dataset"""

    def _imbalanced_data_X_y(problem_type, categorical_columns, size):
        """ "Generates a dummy classification dataset with particular amounts of class imbalance and categorical input columns.
        For our targets, we maintain a 1:5, or 0.2, class ratio of minority : majority.
        We only generate minimum amount for X to set the logical_types, so the length of X and y will be different.

        Args:
            problem_type (str): Either 'binary' or 'multiclass'
            categorical_columns (str): Determines how many categorical cols to use. Either 'all', 'some', or 'none'.
            size (str): Either 'large' or 'small'. 'large' returns a dataset of size 21,000, while 'small' returns a size of 4200
        """
        multiplier = 5 if size == "large" else 1
        col_names = [f"col_{i}" for i in range(100)]
        # generate X to be all int values
        X_dict = {
            col_name: [i % (j + 1) for i in range(1, 100)]
            for j, col_name in enumerate(col_names)
        }
        X = pd.DataFrame(X_dict)
        if categorical_columns == "all":
            X.ww.init(logical_types={col_name: "Categorical" for col_name in col_names})
        elif categorical_columns == "some":
            X.ww.init(
                logical_types={
                    col_name: "Categorical"
                    for col_name in col_names[: len(col_names) // 2]
                },
            )
        else:
            X.ww.init()
        if problem_type == "binary":
            targets = [0] * 3500 + [1] * 700
        else:
            targets = [0] * 3000 + [1] * 600 + [2] * 600
        targets *= multiplier
        y = ww.init_series(pd.Series(targets))
        return X, y

    return _imbalanced_data_X_y


class _AutoMLTestEnv:
    """A test environment that makes it easy to test automl behavior with patched pipeline computations.

    This class provides a context manager that will automatically patch pipeline fit/score/predict_proba methods,
    as well as _encode_targets, BinaryClassificationObjective.optimize_threshold, and skopt.Optimizer.tell. These are
    the most time consuming operations during search, so your test will run as fast as possible.

    This class is ideal for tests that verify some behavior of AutoMLSearch that can be controlled via the side_effect
    or return_value parameters exposed to the patched methods but it may not be suitable for all tests, such as
    tests that patch Estimator.fit instead of Pipeline.fit or tests that only want to patch a selective
    subset of the methods listed above.

    Example:
        >>> env = _AutoMLTestEnv(problem_type="binary")
        >>> # run_search is short-hand for creating the context manager and then running search
        >>> # env.run_search(automl, score_return_value={automl.objective.name: 1.0})
        >>> # with env.test_context(score_return_value={automl.objective.name: 1.0}):
        >>> #    automl.search()
        >>> # env.mock_fit.assert_called_once()
        >>> # env.mock_score.assert_called_once()
    """

    def __init__(self, problem_type):
        """Create a test environment.

        Args:
            problem_type (str): The problem type corresponding to the search class you want to test.

        Attributes:
            mock_fit (MagicMock): MagicMock corresponding to the pipeline.fit method for the latest automl computation.
                Set to None until the first computation is run in the test environment.
            mock_tell (MagicMock): Magic mock corresponding to the skopt.Optimizer.tell method. Set to None unil the
                first computation is run in the test environment.
            mock_score (MagicMock): MagicMock corresponding to the pipeline.score method for the latest automl computation.
                Set to None until the first computation is run in the test environment.
            mock_encode_targets (MagicMock): MagicMock corresponding to the pipeline._encode_targets method for the latest automl computation.
                Set to None until the first computation is run in the test environment.
            mock_predict_proba (MagicMock): MagicMock corresponding to the pipeline.predict_proba method for the latest automl computation.
                Set to None until the first computation is run in the test environment.
            mock_optimize_threshold (MagicMock): MagicMock corresponding to the BinaryClassificationObjective.optimize_threshold for the latest automl computation.
                Set to None until the first computation is run in the test environment.
        """
        self.problem_type = handle_problem_types(problem_type)
        self._mock_fit = None
        self._mock_tell = None
        self._mock_score = None
        self._mock_get_names = None
        self._mock_encode_targets = None
        self._mock_predict_proba = None
        self._mock_predict_proba_in_sample = None
        self._mock_optimize_threshold = None

    @property
    def _pipeline_class(self):
        return {
            ProblemTypes.REGRESSION: "evalml.pipelines.RegressionPipeline",
            ProblemTypes.BINARY: "evalml.pipelines.BinaryClassificationPipeline",
            ProblemTypes.MULTICLASS: "evalml.pipelines.MulticlassClassificationPipeline",
            ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesRegressionPipeline",
            ProblemTypes.TIME_SERIES_MULTICLASS: "evalml.pipelines.TimeSeriesMulticlassClassificationPipeline",
            ProblemTypes.TIME_SERIES_BINARY: "evalml.pipelines.TimeSeriesBinaryClassificationPipeline",
        }[self.problem_type]

    def _patch_method(self, method, side_effect, return_value, pipeline_class_str=None):
        kwargs = {}
        if pipeline_class_str is None:
            pipeline_class_str = self._pipeline_class
        if side_effect is not None:
            kwargs = {"side_effect": side_effect}
        elif return_value is not None:
            kwargs = {"return_value": return_value}
        return patch(pipeline_class_str + "." + method, **kwargs)

    def _reset_mocks(self):
        """Set the mocks to None before running a computation so that we can prevent users from trying to access
        them before leaving the context manager."""
        self._mock_fit = None
        self._mock_tell = None
        self._mock_score = None
        self._mock_get_names = None
        self._mock_encode_targets = None
        self._mock_predict_proba = None
        self._mock_predict_proba_in_sample = None
        self._mock_optimize_threshold = None

    def _get_mock(self, mock_name):
        mock = getattr(self, f"_mock_{mock_name}")
        if mock is None:
            raise ValueError(
                f"mock_{mock_name} cannot be accessed before leaving the test_context! "
                "Access it after leaving test_context.",
            )
        return mock

    @property
    def mock_fit(self):
        return self._get_mock("fit")

    @property
    def mock_tell(self):
        return self._get_mock("tell")

    @property
    def mock_score(self):
        return self._get_mock("score")

    @property
    def mock_encode_targets(self):
        return self._get_mock("encode_targets")

    @property
    def mock_predict_proba(self):
        return self._get_mock("predict_proba")

    @property
    def mock_predict_proba_in_sample(self):
        return self._get_mock("predict_proba_in_sample")

    @property
    def mock_optimize_threshold(self):
        return self._get_mock("optimize_threshold")

    @contextlib.contextmanager
    def test_context(
        self,
        score_return_value=None,
        mock_score_side_effect=None,
        mock_fit_side_effect=None,
        mock_fit_return_value=None,
        predict_proba_return_value=None,
        predict_proba_in_sample_return_value=None,
        optimize_threshold_return_value=0.2,
    ):
        """A context manager for creating an environment that patches time-consuming pipeline methods.
        Sets the mock_fit, mock_score, mock_encode_targets, mock_predict_proba, mock_optimize_threshold attributes.

        Args:
            score_return_value: Passed as the return_value argument of the pipeline.score patch.
            mock_score_side_effect: Passed as the side_effect argument of the pipeline.score patch. Takes precedence over
                score_return_value.
            mock_fit_side_effect: Passed as the side_effect argument of the pipeline.fit patch. Takes precedence over mock_fit_return_value.
            mock_fit_return_value: Passed as the return_value argument of the pipeline.fit patch.
            predict_proba_return_value: Passed as the return_value argument of the pipeline.predict_proba patch.
            optimize_threshold_return_value: Passed as the return value of BinaryClassificationObjective.optimize_threshold patch.
        """
        mock_fit = self._patch_method(
            "fit",
            side_effect=mock_fit_side_effect,
            return_value=mock_fit_return_value,
        )
        mock_score = self._patch_method(
            "score",
            side_effect=mock_score_side_effect,
            return_value=score_return_value,
        )
        mock_get_names = patch(
            "evalml.pipelines.components.FeatureSelector.get_names",
            return_value=[],
        )

        # For simplicity, we will always mock predict_proba and _encode_targets even if the problem is not a
        # classification problem. For regression problems, we'll mock BinaryClassificationPipeline but it doesn't
        # matter which one we mock since those methods won't be called for regression.
        pipeline_to_mock = self._pipeline_class
        if is_regression(self.problem_type):
            pipeline_to_mock = "evalml.pipelines.BinaryClassificationPipeline"

        mock_encode_targets = self._patch_method(
            "_encode_targets",
            side_effect=lambda y: y,
            return_value=None,
            pipeline_class_str=pipeline_to_mock,
        )
        mock_predict_proba = self._patch_method(
            "predict_proba",
            side_effect=None,
            return_value=predict_proba_return_value,
            pipeline_class_str=pipeline_to_mock,
        )
        if handle_problem_types(self.problem_type) in [
            ProblemTypes.TIME_SERIES_BINARY,
            ProblemTypes.TIME_SERIES_MULTICLASS,
        ]:
            mock_predict_proba_in_sample = self._patch_method(
                "predict_proba_in_sample",
                side_effect=None,
                return_value=predict_proba_in_sample_return_value,
                pipeline_class_str=pipeline_to_mock,
            )
        else:
            mock_predict_proba_in_sample = None

        mock_optimize = patch(
            "evalml.objectives.BinaryClassificationObjective.optimize_threshold",
            return_value=optimize_threshold_return_value,
        )

        mock_tell = patch("evalml.tuners.skopt_tuner.Optimizer.tell")

        # Reset the mocks from a previous computation so that ValueError can be properly raised if
        # user tries to access mocks before leaving the context
        self._reset_mocks()

        # Unfortunately, in order to set the MagicMock instances as class attributes we need to use the
        # `with ... ` syntax.
        sleep_time = PropertyMock(return_value=0.00000001)
        mock_sleep = patch(
            "evalml.automl.AutoMLSearch._sleep_time",
            new_callable=sleep_time,
        )
        if mock_predict_proba_in_sample is None:
            with mock_sleep, mock_fit as fit, mock_score as score, mock_get_names as get_names, mock_encode_targets as encode, mock_predict_proba as proba, mock_tell as tell, mock_optimize as optimize:
                # Can think of `yield` as blocking this method until the computation finishes running
                yield
                self._mock_fit = fit
                self._mock_tell = tell
                self._mock_score = score
                self._mock_get_names = get_names
                self._mock_encode_targets = encode
                self._mock_predict_proba = proba
                self._mock_optimize_threshold = optimize
        else:
            with mock_sleep, mock_fit as fit, mock_score as score, mock_get_names as get_names, mock_encode_targets as encode, mock_predict_proba as proba, mock_predict_proba_in_sample as proba_in_sample, mock_tell as tell, mock_optimize as optimize:
                # Can think of `yield` as blocking this method until the computation finishes running
                yield
                self._mock_fit = fit
                self._mock_tell = tell
                self._mock_score = score
                self._mock_get_names = get_names
                self._mock_encode_targets = encode
                self._mock_predict_proba = proba
                self._mock_predict_proba_in_sample = proba_in_sample
                self._mock_optimize_threshold = optimize


@pytest.fixture
def AutoMLTestEnv():
    return _AutoMLTestEnv


@pytest.fixture
def tmpdir(tmp_path):
    dir = py.path.local(tmp_path)
    yield dir
    dir.remove(ignore_errors=True)


@pytest.fixture
def df_with_url_and_email():
    X = pd.DataFrame(
        {
            "categorical": ["a", "b", "b", "a", "c"],
            "numeric": [1, 2, 3, 4, 5],
            "email": [
                "abalone_0@gmail.com",
                "AbaloneRings@yahoo.com",
                "abalone_2@abalone.com",
                "$titanic_data%&@hotmail.com",
                "foo*EMAIL@email.org",
            ],
            "integer": [1, 2, 3, 4, 5],
            "boolean": [True, False, True, False, False],
            "nat_lang": ["natural", "language", "understanding", "is", "difficult"],
            "url": [
                "https://evalml.alteryx.com/en/stable/",
                "https://woodwork.alteryx.com/en/stable/guides/statistical_insights.html",
                "https://twitter.com/AlteryxOSS",
                "https://www.twitter.com/AlteryxOSS",
                "https://www.evalml.alteryx.com/en/stable/demos/text_input.html",
            ],
        },
    )
    X.ww.init(
        logical_types={
            "categorical": "Categorical",
            "numeric": "Double",
            "email": "EmailAddress",
            "boolean": "Boolean",
            "nat_lang": "NaturalLanguage",
            "integer": "Integer",
            "url": "URL",
        },
    )
    return X


def CustomClassificationObjectiveRanges(ranges):
    class CustomClassificationObjectiveRanges(BinaryClassificationObjective):
        """Accuracy score for binary and multiclass classification."""

        name = "Classification Accuracy"
        greater_is_better = True
        score_needs_proba = False
        perfect_score = 1.0
        is_bounded_like_percentage = False
        expected_range = ranges
        problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

        def objective_function(self, y_true, y_predicted, X=None):
            """Not implementing since mocked in our tests."""

    return CustomClassificationObjectiveRanges()


def load_daily_temp_local(n_rows=None):
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    temp_data_path = os.path.join(data_folder_path, "daily-min-temperatures.csv")
    X, y = load_data(
        path=temp_data_path,
        index=None,
        target="Temp",
        n_rows=n_rows,
    )
    missing_date_1 = pd.DataFrame([pd.to_datetime("1984-12-31")], columns=["Date"])
    missing_date_2 = pd.DataFrame([pd.to_datetime("1988-12-31")], columns=["Date"])
    missing_y_1 = pd.Series([14.5], name="Temp")
    missing_y_2 = pd.Series([14.5], name="Temp")

    X = pd.concat(
        [
            X.iloc[:1460],
            missing_date_1,
            X.iloc[1460:2920],
            missing_date_2,
            X.iloc[2920:],
        ],
    ).reset_index(drop=True)
    y = pd.concat(
        [
            y.iloc[:1460],
            missing_y_1,
            y.iloc[1460:2920],
            missing_y_2,
            y.iloc[2920:],
        ],
    ).reset_index(drop=True)
    return X, y


@pytest.fixture
def daily_temp_local():
    X, y = load_daily_temp_local()
    return infer_feature_types(X), infer_feature_types(y)


@pytest.fixture
def dummy_data_check_name():
    return "dummy_data_check_name"


@pytest.fixture
def dummy_data_check_validate_output_warnings():

    return [
        {
            "message": "Data check dummy message",
            "data_check_name": "DataCheck",
            "level": "warning",
            "details": {"columns": None, "rows": None},
            "code": "DATA_CHECK_CODE",
        },
        {
            "message": "Data check dummy message",
            "data_check_name": "DataCheck",
            "level": "warning",
            "details": {"columns": None, "rows": None},
            "code": "DATA_CHECK_CODE",
        },
    ]


@pytest.fixture
def dummy_data_check_validate_output_errors():

    return [
        {
            "message": "Data check dummy message",
            "data_check_name": "DataCheck",
            "level": "error",
            "details": {"columns": None, "rows": None},
            "code": "DATA_CHECK_CODE",
        },
        {
            "message": "Data check dummy message",
            "data_check_name": "DataCheck",
            "level": "error",
            "details": {"columns": None, "rows": None},
            "code": "DATA_CHECK_CODE",
        },
    ]


@pytest.fixture
def imputer_test_data():
    X = pd.DataFrame(
        {
            "dates": pd.date_range("01-01-2022", periods=20),
            "categorical col": pd.Series(
                ["zero", "one", "two", "zero", "two"] * 4,
                dtype="category",
            ),
            "int col": [0, 1, 2, 0, 3] * 4,
            "object col": ["b", "b", "a", "c", "d"] * 4,
            "float col": [0.1, 1.0, 0.0, -2.0, 5.0] * 4,
            "bool col": [True, False, False, True, True] * 4,
            "categorical with nan": pd.Series(
                [np.nan, "1", "0", "0", "3"] * 4,
                dtype="category",
            ),
            "int with nan": [np.nan, 1, 0, 0, 1] * 4,
            "float with nan": [0.3, 1.0, np.nan, -1.0, 0.0] * 4,
            "object with nan": ["b", "b", np.nan, "c", np.nan] * 4,
            "bool col with nan": pd.Series(
                [True, np.nan, False, np.nan, True] * 4,
                dtype="category",
            ),
            "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan] * 4,
            "all nan cat": pd.Series(
                [np.nan, np.nan, np.nan, np.nan, np.nan] * 4,
                dtype="category",
            ),
            "natural language col": pd.Series(
                ["cats are really great", "don't", "believe", "me?", "well..."] * 4,
                dtype="string",
            ),
        },
    )
    X.ww.init(
        logical_types={
            "dates": "datetime",
            "categorical col": "categorical",
            "int col": "integer",
            "object col": "categorical",
            "float col": "double",
            "bool col": "boolean",
            "categorical with nan": "categorical",
            "int with nan": "IntegerNullable",
            "float with nan": "double",
            "object with nan": "categorical",
            "bool col with nan": "BooleanNullable",
            "all nan": "unknown",
            "all nan cat": "categorical",
            "natural language col": "NaturalLanguage",
        },
    )
    return X


@pytest.fixture
def generate_seasonal_data():
    """Function that returns data with a linear trend and a seasonal signal with specified period."""

    def generate_real_data(
        period,
        step=None,
        num_periods=20,
        scale=1,
        seasonal_scale=1,
        trend_degree=1,
        freq_str="D",
        set_time_index=False,
    ):
        X, y = load_weather()
        y = y.set_axis(X["Date"]).asfreq(pd.infer_freq(X["Date"]))
        X = X.set_index("Date").asfreq(pd.infer_freq(X["Date"]))
        return X, y

    def generate_synthetic_data(
        period,
        step=None,
        num_periods=20,
        scale=1,
        seasonal_scale=1,
        trend_degree=1,
        freq_str="D",
        set_time_index=False,
    ):
        """Function to generate a sinusoidal signal with a polynomial trend.

        Args:
            period: The length, in units, of the seasonal signal.
            step:
            num_periods: How many periods of the seasonal signal to generate.
            scale: The relative scale of the trend.  Setting it higher increases
                the comparative strength of the trend.
            seasonal_scale: The relative scale of the sinusoidal seasonality.
                Setting it higher increases the comparative strength of the
                trend.
            trend_degree: The degree of the polynomial trend. 1 = linear, 2 =
                quadratic, 3 = cubic.  Specific functional forms defined
                below.
            freq_str: The pandas frequency string used to define the unit of
                time in the series time index.
            set_time_index: Whether to set the time index with a pandas.
                DatetimeIndex.

        Returns:
            X (pandas.DateFrame): A placeholder feature matrix.
            y (pandas.Series): A synthetic, time series target Series.

        """
        if period is None:
            x = np.arange(0, 1, 0.01)
        elif step is not None:
            freq = 2 * np.pi / period / step
            x = np.arange(0, 1, step)
        else:
            freq = 2 * np.pi / period
            x = np.arange(0, period * num_periods, 1)
        dts = pd.date_range(datetime.today(), periods=len(x), freq=freq_str)
        X = pd.DataFrame({"x": x})
        X = X.set_index(dts)

        if trend_degree == 1:
            y_trend = pd.Series(scale * minmax_scale(x + 2))
        elif trend_degree == 2:
            y_trend = pd.Series(scale * minmax_scale(x**2))
        elif trend_degree == 3:
            y_trend = pd.Series(scale * minmax_scale((x - 5) ** 3 + x**2))
        if period is not None:
            y_seasonal = pd.Series(seasonal_scale * np.sin(freq * x))
        else:
            y_seasonal = pd.Series(np.zeros(len(x)))
        y = y_trend + y_seasonal
        if set_time_index:
            y = y.set_axis(dts)
        return X, y

    def _return_proper_func(real_or_synthetic):
        if real_or_synthetic == "synthetic":
            return generate_synthetic_data
        elif real_or_synthetic == "real":
            return generate_real_data

    return _return_proper_func
