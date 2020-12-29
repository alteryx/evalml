from sklearn.model_selection import KFold, StratifiedKFold

from evalml.objectives import get_objective
from evalml.preprocessing.data_splitters import (
    TimeSeriesSplit,
    TrainingValidationSplit
)
from evalml.problem_types import ProblemTypes, handle_problem_types

_LARGE_DATA_ROW_THRESHOLD = int(1e5)

_LARGE_DATA_PERCENT_VALIDATION = 0.75


def get_default_primary_search_objective(problem_type):
    """Get the default primary search objective for a problem type.

    Arguments:
        problem_type (str or ProblemType): problem type of interest.

    Returns:
        ObjectiveBase: primary objective instance for the problem type.
    """
    problem_type = handle_problem_types(problem_type)
    objective_name = {'binary': 'Log Loss Binary',
                      'multiclass': 'Log Loss Multiclass',
                      'regression': 'R2',
                      'time series regression': 'R2'}[problem_type.value]
    return get_objective(objective_name, return_instance=True)


def make_data_splitter(X, y, problem_type, problem_configuration=None, n_splits=3, shuffle=True, random_state=0):
    """Given the training data and ML problem parameters, compute a data splitting method to use during AutoML search.

    Arguments:
        X (pd.DataFrame, ww.DataTable): The input training data of shape [n_samples, n_features].
        y (pd.Series, ww.DataColumn): The target training data of length [n_samples].
        problem_type (ProblemType): the type of machine learning problem.
        problem_configuration (dict, None): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the gap and max_delay variables.
        n_splits (int, None): the number of CV splits, if applicable. Default 3.
        shuffle (bool): whether or not to shuffle the data before splitting, if applicable. Default True.
        random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

    Returns:
        sklearn.model_selection.BaseCrossValidator: data splitting method.
    """
    problem_type = handle_problem_types(problem_type)
    data_splitter = None
    if problem_type == ProblemTypes.REGRESSION:
        data_splitter = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    elif problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        data_splitter = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    elif problem_type in [ProblemTypes.TIME_SERIES_REGRESSION,
                          ProblemTypes.TIME_SERIES_BINARY,
                          ProblemTypes.TIME_SERIES_MULTICLASS]:
        if not problem_configuration:
            raise ValueError("problem_configuration is required for time series problem types")
        data_splitter = TimeSeriesSplit(n_splits=n_splits, gap=problem_configuration.get('gap'),
                                        max_delay=problem_configuration.get('max_delay'))
    if X.shape[0] > _LARGE_DATA_ROW_THRESHOLD:
        data_splitter = TrainingValidationSplit(test_size=_LARGE_DATA_PERCENT_VALIDATION, shuffle=True)
    return data_splitter
