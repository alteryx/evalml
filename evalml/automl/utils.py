from collections import namedtuple

import pandas as pd
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


def make_data_splitter(X, y, problem_type, problem_configuration=None, n_splits=3, shuffle=True, random_seed=0):
    """Given the training data and ML problem parameters, compute a data splitting method to use during AutoML search.

    Arguments:
        X (ww.DataTable, pd.DataFrame): The input training data of shape [n_samples, n_features].
        y (ww.DataColumn, pd.Series): The target training data of length [n_samples].
        problem_type (ProblemType): The type of machine learning problem.
        problem_configuration (dict, None): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the date_index, gap, and max_delay variables. Defaults to None.
        n_splits (int, None): The number of CV splits, if applicable. Defaults to 3.
        shuffle (bool): Whether or not to shuffle the data before splitting, if applicable. Defaults to True.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        sklearn.model_selection.BaseCrossValidator: Data splitting method.
    """
    random_seed = random_seed
    problem_type = handle_problem_types(problem_type)
    if is_time_series(problem_type):
        if not problem_configuration:
            raise ValueError("problem_configuration is required for time series problem types")
        return TimeSeriesSplit(n_splits=n_splits, gap=problem_configuration.get('gap'),
                               max_delay=problem_configuration.get('max_delay'), date_index=problem_configuration.get('date_index'))
    if X.shape[0] > _LARGE_DATA_ROW_THRESHOLD:
        return TrainingValidationSplit(test_size=_LARGE_DATA_PERCENT_VALIDATION, shuffle=shuffle)
    if problem_type == ProblemTypes.REGRESSION:
        return KFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)
    elif problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        return StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)


def tune_binary_threshold(pipeline, objective, problem_type, X_threshold_tuning, y_threshold_tuning):
    """Tunes the threshold of a binary pipeline to the X and y thresholding data

    Arguments:
        pipeline (Pipeline): Pipeline instance to threshold.
        objective (ObjectiveBase): The objective we want to tune with. If not tuneable and best_pipeline is True, will use F1.
        problem_type (ProblemType): The problem type of the pipeline.
        X_threshold_tuning (ww.DataTable): Features to tune pipeline to.
        y_threshold_tuning (ww.DataColumn): Target data to tune pipeline to.
    """
    if is_binary(problem_type) and objective.is_defined_for_problem_type(problem_type) and objective.can_optimize_threshold:
        pipeline.threshold = 0.5
        if X_threshold_tuning:
            y_predict_proba = pipeline.predict_proba(X_threshold_tuning)
            y_predict_proba = y_predict_proba.iloc[:, 1]
            pipeline.optimize_threshold(X_threshold_tuning, y_threshold_tuning, y_predict_proba, objective)


def check_all_pipeline_names_unique(pipelines):
    """Checks whether all the pipeline names are unique.

    Arguments:
        pipelines (list(PipelineBase)): List of pipelines to check if all names are unique.

    Returns:
          None

    Raises:
        ValueError: if any pipeline names are duplicated.
    """
    name_count = pd.Series([p.name for p in pipelines]).value_counts()
    duplicate_names = name_count[name_count > 1].index.tolist()

    if duplicate_names:
        plural, tense = ("s", "were") if len(duplicate_names) > 1 else ("", "was")
        duplicates = ", ".join([f"'{name}'" for name in sorted(duplicate_names)])
        raise ValueError(f"All pipeline names must be unique. The name{plural} {duplicates} {tense} repeated.")


AutoMLConfig = namedtuple("AutoMLConfig", ["ensembling_indices", "data_splitter", "problem_type",
                                           "objective", "additional_objectives", "optimize_thresholds",
                                           "error_callback", "random_seed"])


def get_best_sampler_for_data(X, y, sampler_type, sampler_balanced_ratio):
    """Returns the name of the sampler component to use for AutoMLSearch.

    Arguments:
        X (ww.DataTable): The input feature data
        y (ww.DataColumn): The input target data
        sampler_type (str): The sampler_type argument passed to AutoMLSearch
        sampler_balanced_ratio (float): The ratio of min:majority targets that we would consider balanced,
            or should balance the classes to.

    Returns:
        str: The string name of the sampling component to use
    """
    # we check for the class balances
    counts = y.to_series().value_counts()
    minority_class = min(counts)
    class_ratios = minority_class / counts
    # if all class ratios are larger than the ratio provided, we don't need to sample
    if all(class_ratios >= sampler_balanced_ratio):
        return None
    # we default to using the Undersampler
    return 'Undersampler'
