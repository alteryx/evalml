"""Utilities useful in AutoML."""
from collections import namedtuple

import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline,
)
from evalml.preprocessing.data_splitters import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    TrainingValidationSplit,
)
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_binary,
    is_time_series,
)
from evalml.utils import import_or_raise

_LARGE_DATA_ROW_THRESHOLD = int(1e5)
_SAMPLER_THRESHOLD = 20000
_LARGE_DATA_PERCENT_VALIDATION = 0.75


def get_default_primary_search_objective(problem_type):
    """Get the default primary search objective for a problem type.

    Args:
        problem_type (str or ProblemType): Problem type of interest.

    Returns:
        ObjectiveBase: primary objective instance for the problem type.
    """
    problem_type = handle_problem_types(problem_type)
    objective_name = {
        "binary": "Log Loss Binary",
        "multiclass": "Log Loss Multiclass",
        "regression": "R2",
        "time series regression": "MedianAE",
        "time series binary": "Log Loss Binary",
        "time series multiclass": "Log Loss Multiclass",
    }[problem_type.value]
    return get_objective(objective_name, return_instance=True)


def make_data_splitter(
    X,
    y,
    problem_type,
    problem_configuration=None,
    n_splits=3,
    shuffle=True,
    random_seed=0,
):
    """Given the training data and ML problem parameters, compute a data splitting method to use during AutoML search.

    Args:
        X (pd.DataFrame): The input training data of shape [n_samples, n_features].
        y (pd.Series): The target training data of length [n_samples].
        problem_type (ProblemType): The type of machine learning problem.
        problem_configuration (dict, None): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the time_index, gap, and max_delay variables. Defaults to None.
        n_splits (int, None): The number of CV splits, if applicable. Defaults to 3.
        shuffle (bool): Whether or not to shuffle the data before splitting, if applicable. Defaults to True.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Returns:
        sklearn.model_selection.BaseCrossValidator: Data splitting method.

    Raises:
        ValueError: If problem_configuration is not given for a time-series problem.
    """
    random_seed = random_seed
    problem_type = handle_problem_types(problem_type)
    if is_time_series(problem_type):
        if not problem_configuration:
            raise ValueError(
                "problem_configuration is required for time series problem types",
            )
        return TimeSeriesSplit(
            n_splits=n_splits,
            gap=problem_configuration.get("gap"),
            max_delay=problem_configuration.get("max_delay"),
            time_index=problem_configuration.get("time_index"),
            forecast_horizon=problem_configuration.get("forecast_horizon"),
        )
    if X.shape[0] > _LARGE_DATA_ROW_THRESHOLD:
        return TrainingValidationSplit(
            test_size=_LARGE_DATA_PERCENT_VALIDATION,
            shuffle=shuffle,
        )
    if problem_type == ProblemTypes.REGRESSION:
        return KFold(n_splits=n_splits, random_state=random_seed, shuffle=shuffle)
    elif problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        return StratifiedKFold(
            n_splits=n_splits,
            random_state=random_seed,
            shuffle=shuffle,
        )


def tune_binary_threshold(
    pipeline,
    objective,
    problem_type,
    X_threshold_tuning,
    y_threshold_tuning,
    X=None,
    y=None,
):
    """Tunes the threshold of a binary pipeline to the X and y thresholding data.

    Args:
        pipeline (Pipeline): Pipeline instance to threshold.
        objective (ObjectiveBase): The objective we want to tune with. If not tuneable and best_pipeline is True, will use F1.
        problem_type (ProblemType): The problem type of the pipeline.
        X_threshold_tuning (pd.DataFrame): Features to which the pipeline will be tuned.
        y_threshold_tuning (pd.Series): Target data to which the pipeline will be tuned.
        X (pd.DataFrame): Features to which the pipeline will be trained (used for time series binary). Defaults to None.
        y (pd.Series): Target to which the pipeline will be trained (used for time series binary). Defaults to None.
    """
    if (
        is_binary(problem_type)
        and objective.is_defined_for_problem_type(problem_type)
        and objective.can_optimize_threshold
    ):
        pipeline.threshold = 0.5
        if X_threshold_tuning is not None:
            if problem_type == ProblemTypes.TIME_SERIES_BINARY:
                y_predict_proba = pipeline.predict_proba_in_sample(
                    X_threshold_tuning,
                    y_threshold_tuning,
                    X,
                    y,
                )
            else:
                y_predict_proba = pipeline.predict_proba(X_threshold_tuning, X, y)
            y_predict_proba = y_predict_proba.iloc[:, 1]
            pipeline.optimize_threshold(
                X_threshold_tuning,
                y_threshold_tuning,
                y_predict_proba,
                objective,
            )


def check_all_pipeline_names_unique(pipelines):
    """Checks whether all the pipeline names are unique.

    Args:
        pipelines (list[PipelineBase]): List of pipelines to check if all names are unique.

    Raises:
        ValueError: If any pipeline names are duplicated.
    """
    name_count = pd.Series([p.name for p in pipelines]).value_counts()
    duplicate_names = name_count[name_count > 1].index.tolist()

    if duplicate_names:
        plural, tense = ("s", "were") if len(duplicate_names) > 1 else ("", "was")
        duplicates = ", ".join([f"'{name}'" for name in sorted(duplicate_names)])
        raise ValueError(
            f"All pipeline names must be unique. The name{plural} {duplicates} {tense} repeated.",
        )


AutoMLConfig = namedtuple(
    "AutoMLConfig",
    [
        "data_splitter",
        "problem_type",
        "objective",
        "additional_objectives",
        "alternate_thresholding_objective",
        "optimize_thresholds",
        "error_callback",
        "random_seed",
        "X_schema",
        "y_schema",
        "errors",
    ],
)


def get_best_sampler_for_data(X, y, sampler_method, sampler_balanced_ratio):
    """Returns the name of the sampler component to use for AutoMLSearch.

    Args:
        X (pd.DataFrame): The input feature data
        y (pd.Series): The input target data
        sampler_method (str): The sampler_type argument passed to AutoMLSearch
        sampler_balanced_ratio (float): The ratio of min:majority targets that we would consider balanced,
            or should balance the classes to.

    Returns:
        str, None: The string name of the sampling component to use, or None if no sampler is necessary
    """
    # we check for the class balances
    counts = y.value_counts()
    minority_class = min(counts)
    class_ratios = minority_class / counts
    # if all class ratios are larger than the ratio provided, we don't need to sample
    if all(class_ratios >= sampler_balanced_ratio):
        return None
    # We set a threshold to use the Undersampler in order to avoid long runtimes
    elif len(y) >= _SAMPLER_THRESHOLD and sampler_method != "Oversampler":
        return "Undersampler"
    else:
        try:
            import_or_raise(
                "imblearn.over_sampling",
                error_msg="imbalanced-learn is not installed",
            )
            return "Oversampler"
        except ImportError:
            return "Undersampler"


def get_pipelines_from_component_graphs(
    component_graphs_dict,
    problem_type,
    parameters=None,
    random_seed=0,
):
    """Returns created pipelines from passed component graphs based on the specified problem type.

    Args:
        component_graphs_dict (dict): The dict of component graphs.
        problem_type (str or ProblemType): The problem type for which pipelines will be created.
        parameters (dict): Pipeline-level parameters that should be passed to the proposed pipelines. Defaults to None.
        random_seed (int): Random seed. Defaults to 0.

    Returns:
        list: List of pipelines made from the passed component graphs.
    """
    pipeline_class = {
        ProblemTypes.BINARY: BinaryClassificationPipeline,
        ProblemTypes.MULTICLASS: MulticlassClassificationPipeline,
        ProblemTypes.REGRESSION: RegressionPipeline,
        ProblemTypes.TIME_SERIES_BINARY: TimeSeriesBinaryClassificationPipeline,
        ProblemTypes.TIME_SERIES_MULTICLASS: TimeSeriesMulticlassClassificationPipeline,
        ProblemTypes.TIME_SERIES_REGRESSION: TimeSeriesRegressionPipeline,
    }[handle_problem_types(problem_type)]
    created_pipelines = []
    for graph_name, component_graph in component_graphs_dict.items():
        created_pipelines.append(
            pipeline_class(
                component_graph=component_graph,
                parameters=parameters,
                custom_name=graph_name,
                random_seed=random_seed,
            ),
        )
    return created_pipelines
