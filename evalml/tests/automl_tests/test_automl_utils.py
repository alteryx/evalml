import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from evalml.automl.utils import (
    _LARGE_DATA_PERCENT_VALIDATION,
    _LARGE_DATA_ROW_THRESHOLD,
    get_default_primary_search_objective,
    make_data_splitter
)
from evalml.objectives import R2, LogLossBinary, LogLossMulticlass
from evalml.preprocessing.data_splitters import (
    TimeSeriesSplit,
    TrainingValidationSplit
)
from evalml.problem_types import ProblemTypes


def test_get_default_primary_search_objective():
    assert isinstance(get_default_primary_search_objective("binary"), LogLossBinary)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.BINARY), LogLossBinary)
    assert isinstance(get_default_primary_search_objective("multiclass"), LogLossMulticlass)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.MULTICLASS), LogLossMulticlass)
    assert isinstance(get_default_primary_search_objective("regression"), R2)
    assert isinstance(get_default_primary_search_objective(ProblemTypes.REGRESSION), R2)
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
    if large_data:
        assert isinstance(data_splitter, TrainingValidationSplit)
        assert data_splitter.shuffle
        assert data_splitter.test_size == _LARGE_DATA_PERCENT_VALIDATION
        assert data_splitter.stratify is None
        assert data_splitter.random_state == 0
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

    if problem_type in [ProblemTypes.TIME_SERIES_REGRESSION,
                        ProblemTypes.TIME_SERIES_BINARY,
                        ProblemTypes.TIME_SERIES_MULTICLASS]:
        assert isinstance(data_splitter, TimeSeriesSplit)
        assert data_splitter.n_splits == 3
        assert data_splitter.gap == 1
        assert data_splitter.max_delay == 7


def test_make_data_splitter_parameters():
    n = 10
    X = pd.DataFrame({'col_0': list(range(n)),
                      'target': list(range(n))})
    y = X.pop('target')
    random_state = 42

    data_splitter = make_data_splitter(X, y, ProblemTypes.REGRESSION, n_splits=5, shuffle=False, random_state=random_state)
    assert isinstance(data_splitter, KFold)
    assert data_splitter.n_splits == 5
    assert not data_splitter.shuffle
    assert data_splitter.random_state == random_state

    data_splitter = make_data_splitter(X, y, ProblemTypes.BINARY, n_splits=5, shuffle=False, random_state=random_state)
    assert isinstance(data_splitter, StratifiedKFold)
    assert data_splitter.n_splits == 5
    assert not data_splitter.shuffle
    assert data_splitter.random_state == random_state

    data_splitter = make_data_splitter(X, y, ProblemTypes.MULTICLASS, n_splits=5, shuffle=False, random_state=random_state)
    assert isinstance(data_splitter, StratifiedKFold)
    assert data_splitter.n_splits == 5
    assert not data_splitter.shuffle
    assert data_splitter.random_state == random_state

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
        make_data_splitter(X, y, ProblemTypes.TIME_SERIES_REGRESSION, n_splits=5, shuffle=False)
    with pytest.raises(KeyError, match="Problem type 'XYZ' does not exist"):
        make_data_splitter(X, y, 'XYZ', n_splits=5, shuffle=False)
