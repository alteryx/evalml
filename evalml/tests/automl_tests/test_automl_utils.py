import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from evalml.automl.utils import make_data_splitter

from evalml import AutoMLSearch
from evalml.automl.data_splitters import (
    TimeSeriesSplit,
    TrainingValidationSplit
)
from evalml.problem_types import ProblemTypes


def test_make_data_splitter():
    X = pd.DataFrame({'col_0': list(range(10)),
                      'target': list(range(10))})
    y = X.pop('target')
    data_splitter = make_data_splitter(X, y, ProblemTypes.REGRESSION)
    assert isinstance(data_splitter, KFold)
    assert data_splitter.n_splits == 3
