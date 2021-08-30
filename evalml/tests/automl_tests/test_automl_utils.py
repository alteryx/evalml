from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from evalml.automl.utils import (
    _LARGE_DATA_PERCENT_VALIDATION,
    _LARGE_DATA_ROW_THRESHOLD,
    get_best_sampler_for_data,
    get_default_primary_search_objective,
    get_pipelines_from_component_graphs,
    make_data_splitter,
    tune_binary_threshold,
)
from evalml.objectives import F1, R2, LogLossBinary, LogLossMulticlass
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
)
from evalml.preprocessing.data_splitters import (
    TimeSeriesSplit,
    TrainingValidationSplit,
)
from evalml.problem_types import ProblemTypes
from evalml.utils.woodwork_utils import infer_feature_types


def test_get_default_primary_search_objective():
    assert isinstance(get_default_primary_search_objective("binary"), LogLossBinary)
    assert isinstance(
        get_default_primary_search_objective(ProblemTypes.BINARY), LogLossBinary
    )
    assert isinstance(
        get_default_primary_search_objective("multiclass"), LogLossMulticlass
    )
    assert isinstance(
        get_default_primary_search_objective(ProblemTypes.MULTICLASS), LogLossMulticlass
    )
    assert isinstance(get_default_primary_search_objective("regression"), R2)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.REGRESSION), R2)
    assert isinstance(
        get_default_primary_search_objective("time series binary"), LogLossBinary
    )
    assert isinstance(
        get_default_primary_search_objective(ProblemTypes.TIME_SERIES_BINARY),
        LogLossBinary,
    )
    assert isinstance(
        get_default_primary_search_objective("time series multiclass"),
        LogLossMulticlass,
    )
    assert isinstance(
        get_default_primary_search_objective(ProblemTypes.TIME_SERIES_MULTICLASS),
        LogLossMulticlass,
    )
    with pytest.raises(KeyError, match="Problem type 'auto' does not exist"):
        get_default_primary_search_objective("auto")


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize("large_data", [False, True])
def test_make_data_splitter_default(problem_type, large_data):
    n = 10
    if large_data:
        n = _LARGE_DATA_ROW_THRESHOLD + 1
    X = pd.DataFrame({"col_0": list(range(n)), "target": list(range(n))})
    y = X.pop("target")

    problem_configuration = None
    if problem_type in [
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]:
        problem_configuration = {"gap": 1, "max_delay": 7, "date_index": None}

    data_splitter = make_data_splitter(
        X, y, problem_type, problem_configuration=problem_configuration
    )
    if large_data and problem_type in [
        ProblemTypes.REGRESSION,
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
    ]:
        assert isinstance(data_splitter, TrainingValidationSplit)
        assert data_splitter.stratify is None
        assert data_splitter.random_seed == 0
        assert data_splitter.shuffle
        assert data_splitter.test_size == _LARGE_DATA_PERCENT_VALIDATION
        return

    if problem_type == ProblemTypes.REGRESSION:
        assert isinstance(data_splitter, KFold)
        assert data_splitter.n_splits == 3
        assert data_splitter.shuffle
        assert data_splitter.random_state == 0

    if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        assert isinstance(data_splitter, StratifiedKFold)
        assert data_splitter.n_splits == 3
        assert data_splitter.shuffle
        assert data_splitter.random_state == 0

    if problem_type in [
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]:
        assert isinstance(data_splitter, TimeSeriesSplit)
        assert data_splitter.n_splits == 3
        assert data_splitter.gap == 1
        assert data_splitter.max_delay == 7
        assert data_splitter.date_index is None


@pytest.mark.parametrize(
    "problem_type, expected_data_splitter",
    [
        (ProblemTypes.REGRESSION, KFold),
        (ProblemTypes.BINARY, StratifiedKFold),
        (ProblemTypes.MULTICLASS, StratifiedKFold),
    ],
)
def test_make_data_splitter_parameters(problem_type, expected_data_splitter):
    n = 10
    X = pd.DataFrame({"col_0": list(range(n)), "target": list(range(n))})
    y = X.pop("target")
    random_seed = 42

    data_splitter = make_data_splitter(
        X, y, problem_type, n_splits=5, random_seed=random_seed
    )
    assert isinstance(data_splitter, expected_data_splitter)
    assert data_splitter.n_splits == 5
    assert data_splitter.shuffle
    assert data_splitter.random_state == random_seed


def test_make_data_splitter_parameters_time_series():
    n = 10
    X = pd.DataFrame({"col_0": list(range(n)), "target": list(range(n))})
    y = X.pop("target")

    for problem_type in [
        ProblemTypes.TIME_SERIES_REGRESSION,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]:
        data_splitter = make_data_splitter(
            X,
            y,
            problem_type,
            problem_configuration={"gap": 1, "max_delay": 7, "date_index": None},
            n_splits=5,
            shuffle=False,
        )
        assert isinstance(data_splitter, TimeSeriesSplit)
        assert data_splitter.n_splits == 5
        assert data_splitter.gap == 1
        assert data_splitter.max_delay == 7
        assert data_splitter.date_index is None


def test_make_data_splitter_error():
    n = 10
    X = pd.DataFrame({"col_0": list(range(n)), "target": list(range(n))})
    y = X.pop("target")

    with pytest.raises(
        ValueError,
        match="problem_configuration is required for time series problem types",
    ):
        make_data_splitter(X, y, ProblemTypes.TIME_SERIES_REGRESSION)
    with pytest.raises(KeyError, match="Problem type 'XYZ' does not exist"):
        make_data_splitter(X, y, "XYZ")


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS],
)
@pytest.mark.parametrize("large_data", [True, False])
def test_make_data_splitter_error_shuffle_random_state(problem_type, large_data):
    n = 10
    if large_data:
        n = _LARGE_DATA_ROW_THRESHOLD + 1
    X = pd.DataFrame({"col_0": list(range(n)), "target": list(range(n))})
    y = X.pop("target")

    if large_data:
        make_data_splitter(
            X, y, problem_type, n_splits=5, shuffle=False, random_seed=42
        )
    else:
        with pytest.raises(
            ValueError,
            match="Setting a random_state has no effect since shuffle is False.",
        ):
            make_data_splitter(
                X, y, problem_type, n_splits=5, shuffle=False, random_seed=42
            )


@patch("evalml.objectives.BinaryClassificationObjective.optimize_threshold")
@patch(
    "evalml.pipelines.BinaryClassificationPipeline._encode_targets",
    side_effect=lambda y: y,
)
@patch("evalml.pipelines.BinaryClassificationPipeline.predict_proba")
@patch("evalml.pipelines.BinaryClassificationPipeline.score")
@patch("evalml.pipelines.BinaryClassificationPipeline.fit")
def test_tune_binary_threshold(
    mock_fit,
    mock_score,
    mock_predict_proba,
    mock_encode_targets,
    mock_optimize_threshold,
    dummy_binary_pipeline_class,
    X_y_binary,
):
    mock_optimize_threshold.return_value = 0.42
    mock_score.return_value = {"F1": 1.0}
    X, y = X_y_binary
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    pipeline = dummy_binary_pipeline_class({})
    tune_binary_threshold(pipeline, F1(), "binary", X, y)
    assert pipeline.threshold == 0.42

    pipeline = dummy_binary_pipeline_class({})
    tune_binary_threshold(pipeline, F1(), "binary", None, None)
    assert pipeline.threshold == 0.5

    pipeline = dummy_binary_pipeline_class({})
    tune_binary_threshold(pipeline, F1(), "multiclass", X, y)
    assert pipeline.threshold is None


@pytest.mark.parametrize("size", ["large", "small"])
@pytest.mark.parametrize("categorical_columns", ["none", "all", "some"])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
@pytest.mark.parametrize("sampler_balanced_ratio", [1, 0.5, 0.25, 0.2, 0.1, 0.05])
def test_get_best_sampler_for_data_auto(
    sampler_balanced_ratio,
    problem_type,
    categorical_columns,
    size,
    mock_imbalanced_data_X_y,
    has_minimal_dependencies,
):
    X, y = mock_imbalanced_data_X_y(problem_type, categorical_columns, size)
    name_output = get_best_sampler_for_data(X, y, "auto", sampler_balanced_ratio)
    if sampler_balanced_ratio <= 0.2:
        # the imbalanced data we get has a class ratio of 0.2 minority:majority
        assert name_output is None
    else:
        if size == "large" or has_minimal_dependencies:
            assert name_output == "Undersampler"
        else:
            assert name_output == "Oversampler"


@pytest.mark.parametrize("sampler_method", ["Undersampler", "Oversampler"])
@pytest.mark.parametrize("categorical_columns", ["none", "all", "some"])
def test_get_best_sampler_for_data_sampler_method(
    categorical_columns,
    sampler_method,
    mock_imbalanced_data_X_y,
    has_minimal_dependencies,
):
    X, y = mock_imbalanced_data_X_y("binary", categorical_columns, "large")
    name_output = get_best_sampler_for_data(X, y, sampler_method, 0.5)
    if sampler_method == "Undersampler" or has_minimal_dependencies:
        assert name_output == "Undersampler"
    else:
        assert name_output == "Oversampler"


def test_get_best_sampler_for_data_nonnumeric_noncategorical_columns(X_y_binary):
    pytest.importorskip(
        "imblearn.over_sampling",
        reason="Skipping oversampling test because imbalanced-learn is not installed",
    )
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series([i % 5 == 0 for i in range(100)])
    X[0] = [i % 2 for i in range(100)]
    X_ww = infer_feature_types(X, feature_types={0: "boolean", 1: "categorical"})

    name_output = get_best_sampler_for_data(X_ww, y, "Oversampler", 0.8)
    assert name_output == "Oversampler"

    X = X.drop([i for i in range(2, 20)], axis=1)  # remove all numeric columns
    X_ww = infer_feature_types(X, feature_types={0: "boolean", 1: "categorical"})
    name_output = get_best_sampler_for_data(X_ww, y, "Oversampler", 0.5)
    assert name_output == "Oversampler"

    X_ww = infer_feature_types(X, feature_types={0: "boolean", 1: "boolean"})
    name_output = get_best_sampler_for_data(X_ww, y, "Oversampler", 0.5)
    assert name_output == "Oversampler"


@pytest.mark.parametrize(
    "problem_type,estimator",
    [
        ("binary", "Random Forest Classifier"),
        ("multiclass", "Random Forest Classifier"),
        ("regression", "Random Forest Regressor"),
        ("time series regression", "ARIMA Regressor"),
    ],
)
def test_get_pipelines_from_component_graphs(problem_type, estimator):
    component_graphs = {
        "Name_0": {
            "Imputer": ["Imputer", "X", "y"],
            "Imputer_1": ["Imputer", "Imputer.x", "y"],
            estimator: [estimator, "Imputer_1.x", "y"],
        },
        "Name_1": {
            "Imputer": ["Imputer", "X", "y"],
            estimator: [estimator, "Imputer.x", "y"],
        },
    }
    if problem_type == "time series regression":
        with pytest.raises(ValueError, match="date_index, gap, and max_delay"):
            get_pipelines_from_component_graphs(component_graphs, problem_type)
    else:
        returned_pipelines = get_pipelines_from_component_graphs(
            component_graphs, problem_type
        )
        assert returned_pipelines[0].random_seed == 0
        assert returned_pipelines[1].random_seed == 0
        if problem_type == "binary":
            assert all(
                isinstance(pipe_, BinaryClassificationPipeline)
                for pipe_ in returned_pipelines
            )
        elif problem_type == "multiclass":
            assert all(
                isinstance(pipe_, MulticlassClassificationPipeline)
                for pipe_ in returned_pipelines
            )
        elif problem_type == "regression":
            assert all(
                isinstance(pipe_, RegressionPipeline) for pipe_ in returned_pipelines
            )
