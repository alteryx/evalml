import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from evalml.automl.utils import make_data_splitter, _LARGE_DATA_ROW_THRESHOLD, _LARGE_DATA_PERCENT_VALIDATION

from evalml.automl.data_splitters import (
    TimeSeriesSplit,
    TrainingValidationSplit
)
from evalml.problem_types import ProblemTypes


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
    if problem_type in [ProblemTypes.TIME_SERIES_REGRESSION]:
        problem_configuration = {'gap': 1, 'max_delay': 7}

    data_splitter = make_data_splitter(X, y, problem_type, problem_configuration=problem_configuration)
    if large_data:
        assert isinstance(data_splitter, TrainingValidationSplit)
        assert data_splitter.shuffle
        assert data_splitter.test_size == _LARGE_DATA_PERCENT_VALIDATION
        assert data_splitter.stratify == (problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
        return

    if problem_type == ProblemTypes.REGRESSION:
        assert isinstance(data_splitter, KFold)
        assert data_splitter.n_splits == 3
        assert data_splitter.shuffle

    if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        assert isinstance(data_splitter, StratifiedKFold)
        assert data_splitter.n_splits == 3
        assert data_splitter.shuffle

    if problem_type in [ProblemTypes.TIME_SERIES_REGRESSION]:
        assert isinstance(data_splitter, TimeSeriesSplit)
        assert data_splitter.n_splits == 3
        assert data_splitter.gap == 1
        assert data_splitter.max_delay == 7
