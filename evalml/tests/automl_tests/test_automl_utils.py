from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.model_selection import KFold

from evalml.automl.utils import (
    _LARGE_DATA_PERCENT_VALIDATION,
    _LARGE_DATA_ROW_THRESHOLD,
    get_default_primary_search_objective,
    make_data_splitter,
    tune_binary_threshold
)
from evalml.objectives import F1, R2, LogLossBinary, LogLossMulticlass
from evalml.preprocessing.data_splitters import (
    BalancedClassificationDataCVSplit,
    BalancedClassificationDataTVSplit,
    TimeSeriesSplit,
    TrainingValidationSplit
)
from evalml.problem_types import ProblemTypes
from evalml.utils.woodwork_utils import infer_feature_types


def test_get_default_primary_search_objective():
    assert isinstance(get_default_primary_search_objective("binary"), LogLossBinary)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.BINARY), LogLossBinary)
    assert isinstance(get_default_primary_search_objective("multiclass"), LogLossMulticlass)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.MULTICLASS), LogLossMulticlass)
    assert isinstance(get_default_primary_search_objective("regression"), R2)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.REGRESSION), R2)
    assert isinstance(get_default_primary_search_objective("time series binary"), LogLossBinary)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.TIME_SERIES_BINARY), LogLossBinary)
    assert isinstance(get_default_primary_search_objective("time series multiclass"), LogLossMulticlass)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.TIME_SERIES_MULTICLASS), LogLossMulticlass)
    with pytest.raises(KeyError, match="Problem type 'auto' does not exist"):
        get_default_primary_search_objective("auto")


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize("large_data", [False, True])
def test_make_data_splitter_default(problem_type, large_data):
    n = 10
    if large_data:
        n = _LARGE_DATA_ROW_THRESHOLD + 1
    X = pd.DataFrame({'col_0': list(range(n)),
                      'target': list(range(n))})
    y = X.pop('target')

    problem_configuration = None
    if problem_type in [ProblemTypes.TIME_SERIES_REGRESSION,
                        ProblemTypes.TIME_SERIES_BINARY,
                        ProblemTypes.TIME_SERIES_MULTICLASS]:
        problem_configuration = {'gap': 1, 'max_delay': 7}

    data_splitter = make_data_splitter(X, y, problem_type, problem_configuration=problem_configuration)
    if large_data and problem_type in [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        if problem_type == ProblemTypes.REGRESSION:
            assert isinstance(data_splitter, TrainingValidationSplit)
            assert data_splitter.stratify is None
            assert data_splitter.random_seed == 0
        else:
            assert isinstance(data_splitter, BalancedClassificationDataTVSplit)
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
        assert isinstance(data_splitter, BalancedClassificationDataCVSplit)
        assert data_splitter.n_splits == 3
        assert data_splitter.shuffle
        assert data_splitter.random_seed == 0

    if problem_type in [ProblemTypes.TIME_SERIES_REGRESSION,
                        ProblemTypes.TIME_SERIES_BINARY,
                        ProblemTypes.TIME_SERIES_MULTICLASS]:
        assert isinstance(data_splitter, TimeSeriesSplit)
        assert data_splitter.n_splits == 3
        assert data_splitter.gap == 1
        assert data_splitter.max_delay == 7


@pytest.mark.parametrize("problem_type, expected_data_splitter", [(ProblemTypes.REGRESSION, KFold),
                                                                  (ProblemTypes.BINARY, BalancedClassificationDataCVSplit),
                                                                  (ProblemTypes.MULTICLASS, BalancedClassificationDataCVSplit)])
def test_make_data_splitter_parameters(problem_type, expected_data_splitter):
    n = 10
    X = pd.DataFrame({'col_0': list(range(n)),
                      'target': list(range(n))})
    y = X.pop('target')
    random_seed = 42

    data_splitter = make_data_splitter(X, y, problem_type, n_splits=5, random_seed=random_seed)
    assert isinstance(data_splitter, expected_data_splitter)
    assert data_splitter.n_splits == 5
    assert data_splitter.shuffle
    if str(problem_type) == 'regression':
        assert data_splitter.random_state == random_seed
    else:
        assert data_splitter.random_seed == random_seed


def test_make_data_splitter_parameters_time_series():
    n = 10
    X = pd.DataFrame({'col_0': list(range(n)),
                      'target': list(range(n))})
    y = X.pop('target')

    for problem_type in [ProblemTypes.TIME_SERIES_REGRESSION, ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]:
        data_splitter = make_data_splitter(X, y, problem_type, problem_configuration={'gap': 1, 'max_delay': 7}, n_splits=5, shuffle=False)
        assert isinstance(data_splitter, TimeSeriesSplit)
        assert data_splitter.n_splits == 5
        assert data_splitter.gap == 1
        assert data_splitter.max_delay == 7


def test_make_data_splitter_error():
    n = 10
    X = pd.DataFrame({'col_0': list(range(n)),
                      'target': list(range(n))})
    y = X.pop('target')

    with pytest.raises(ValueError, match="problem_configuration is required for time series problem types"):
        make_data_splitter(X, y, ProblemTypes.TIME_SERIES_REGRESSION)
    with pytest.raises(KeyError, match="Problem type 'XYZ' does not exist"):
        make_data_splitter(X, y, 'XYZ')


@pytest.mark.parametrize("problem_type", [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@pytest.mark.parametrize("large_data", [True, False])
def test_make_data_splitter_error_shuffle_random_state(problem_type, large_data):
    n = 10
    if large_data:
        n = _LARGE_DATA_ROW_THRESHOLD + 1
    X = pd.DataFrame({'col_0': list(range(n)),
                      'target': list(range(n))})
    y = X.pop('target')

    if large_data:
        make_data_splitter(X, y, problem_type, n_splits=5, shuffle=False, random_seed=42)
    else:
        with pytest.raises(ValueError, match="Setting a random_state has no effect since shuffle is False."):
            make_data_splitter(X, y, problem_type, n_splits=5, shuffle=False, random_seed=42)


@patch('evalml.objectives.BinaryClassificationObjective.optimize_threshold')
@patch('evalml.pipelines.BinaryClassificationPipeline._encode_targets', side_effect=lambda y: y)
@patch('evalml.pipelines.BinaryClassificationPipeline.predict_proba')
@patch('evalml.pipelines.BinaryClassificationPipeline.score')
@patch('evalml.pipelines.BinaryClassificationPipeline.fit')
def test_tune_binary_threshold(mock_fit, mock_score, mock_predict_proba, mock_encode_targets, mock_optimize_threshold,
                               dummy_binary_pipeline_class, X_y_binary):
    mock_optimize_threshold.return_value = 0.42
    mock_score.return_value = {'F1': 1.0}
    X, y = X_y_binary
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    pipeline = dummy_binary_pipeline_class({})
    tune_binary_threshold(pipeline, F1(), 'binary', X, y)
    assert pipeline.threshold == 0.42

    pipeline = dummy_binary_pipeline_class({})
    tune_binary_threshold(pipeline, F1(), 'binary', None, None)
    assert pipeline.threshold == 0.5

    pipeline = dummy_binary_pipeline_class({})
    tune_binary_threshold(pipeline, F1(), 'multiclass', X, y)
    assert pipeline.threshold is None
