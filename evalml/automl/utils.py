from sklearn.model_selection import KFold, StratifiedKFold

from evalml.objectives import get_objective
from evalml.preprocessing.data_splitters import (
    TimeSeriesSplit,
    TrainingValidationSplit
)
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_binary,
    is_time_series
)
from evalml.utils import deprecate_arg

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
                      'time series regression': 'R2',
                      'time series binary': 'Log Loss Binary',
                      'time series multiclass': 'Log Loss Multiclass'}[problem_type.value]
    return get_objective(objective_name, return_instance=True)


def make_data_splitter(X, y, problem_type, problem_configuration=None, n_splits=3, shuffle=True,
                       random_state=None, random_seed=0):
    """Given the training data and ML problem parameters, compute a data splitting method to use during AutoML search.

    Arguments:
        X (ww.DataTable, pd.DataFrame): The input training data of shape [n_samples, n_features].
        y (ww.DataColumn, pd.Series): The target training data of length [n_samples].
        problem_type (ProblemType): The type of machine learning problem.
        problem_configuration (dict, None): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the gap and max_delay variables. Defaults to None.
        n_splits (int, None): The number of CV splits, if applicable. Defaults to 3.
        shuffle (bool): Whether or not to shuffle the data before splitting, if applicable. Defaults to True.
        random_state (None, int): Deprecated - use random_seed instead.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        sklearn.model_selection.BaseCrossValidator: Data splitting method.
    """
    random_seed = deprecate_arg("random_state", "random_seed", random_state, random_seed)
    problem_type = handle_problem_types(problem_type)
    if X.shape[0] > _LARGE_DATA_ROW_THRESHOLD:
        return TrainingValidationSplit(test_size=_LARGE_DATA_PERCENT_VALIDATION, shuffle=True)

    if problem_type == ProblemTypes.REGRESSION:
        return KFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)
    elif problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        return StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)
    elif is_time_series(problem_type):
        if not problem_configuration:
            raise ValueError("problem_configuration is required for time series problem types")
        return TimeSeriesSplit(n_splits=n_splits, gap=problem_configuration.get('gap'),
                               max_delay=problem_configuration.get('max_delay'))


def tune_binary_threshold(pipeline, objective, problem_type, X_threshold_tuning, y_threshold_tuning):
    """Tunes the threshold of a binary pipeline to the X and y thresholding data

    Arguments:
        pipeline (Pipeline): Pipeline instance to threshold
        X_threshold_tuning (ww.DataTable): Features to tune pipeline to
        y_threshold_tuning (ww.DataColumn): Target data to tune pipeline to
    """
    if is_binary(problem_type) and objective.is_defined_for_problem_type(problem_type) and objective.can_optimize_threshold:
        pipeline.threshold = 0.5
        if X_threshold_tuning:
            y_predict_proba = pipeline.predict_proba(X_threshold_tuning)
            y_predict_proba = y_predict_proba.iloc[:, 1]
            pipeline.threshold = objective.optimize_threshold(y_predict_proba, y_threshold_tuning, X=X_threshold_tuning)
