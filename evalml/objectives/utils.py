"""Utility methods for EvalML objectives."""
from evalml import objectives
from evalml.exceptions import ObjectiveCreationError, ObjectiveNotFoundError
from evalml.objectives.objective_base import ObjectiveBase
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils.gen_utils import _get_subclasses

DEFAULT_RECOMMENDATION_OBJECTIVES = [
    objectives.F1,
    objectives.BalancedAccuracyBinary,
    objectives.AUC,
    objectives.LogLossBinary,
    objectives.F1Macro,
    objectives.BalancedAccuracyMulticlass,
    objectives.LogLossMulticlass,
    objectives.AUCMicro,
    objectives.MSE,
    objectives.MAE,
    objectives.R2,
    objectives.MedianAE,
]


def get_non_core_objectives():
    """Get non-core objective classes.

    Non-core objectives are objectives that are domain-specific. Users typically need to configure these objectives
    before using them in AutoMLSearch.

    Returns:
        List of ObjectiveBase classes
    """
    return [
        objectives.CostBenefitMatrix,
        objectives.FraudCost,
        objectives.LeadScoring,
        objectives.Recall,
        objectives.RecallMacro,
        objectives.RecallMicro,
        objectives.RecallWeighted,
        objectives.MAPE,
        objectives.MeanSquaredLogError,
        objectives.RootMeanSquaredLogError,
        objectives.SensitivityLowAlert,
    ]


def ranking_only_objectives():
    """Get ranking-only objective classes.

    Ranking-only objectives are objectives that are useful for evaluating the performance of a model, but should not
    be used as an optimization objective during AutoMLSearch for various reasons.

    Returns:
        List of ObjectiveBase classes
    """
    return [
        objectives.Recall,
        objectives.RecallMacro,
        objectives.RecallMicro,
        objectives.RecallWeighted,
        objectives.MAPE,
        objectives.MeanSquaredLogError,
        objectives.RootMeanSquaredLogError,
    ]


def get_optimization_objectives(problem_type):
    """Get objectives for optimization.

    Args:
        problem_type (str/ProblemTypes): Type of problem

    Returns:
        List of ObjectiveBase instances
    """
    problem_type = handle_problem_types(problem_type)
    ranking_only = ranking_only_objectives()
    objectives = [
        obj
        for obj in get_ranking_objectives(problem_type)
        if obj.__class__ not in ranking_only
    ]
    return objectives


def get_ranking_objectives(problem_type):
    """Get objectives for pipeline rankings.

    Args:
        problem_type (str/ProblemTypes): Type of problem

    Returns:
        List of ObjectiveBase instances
    """
    problem_type = handle_problem_types(problem_type)
    all_objectives_dict = _all_objectives_dict()
    objectives = [
        obj()
        for obj in all_objectives_dict.values()
        if obj.is_defined_for_problem_type(problem_type)
        and (obj not in get_non_core_objectives() or obj in ranking_only_objectives())
    ]
    return objectives


def _all_objectives_dict():
    all_objectives = _get_subclasses(ObjectiveBase)
    objectives_dict = {}
    for objective in all_objectives:
        if "evalml.objectives" not in objective.__module__:
            continue
        objectives_dict[objective.name.lower()] = objective
    return objectives_dict


def get_all_objective_names():
    """Get a list of the names of all objectives.

    Returns:
        list (str): Objective names
    """
    all_objectives_dict = _all_objectives_dict()
    return list(all_objectives_dict.keys())


def get_core_objective_names():
    """Get a list of all valid core objectives.

    Returns:
        list[str]: Objective names.
    """
    all_objectives = _all_objectives_dict()
    non_core = get_non_core_objectives()
    return [
        name
        for name, class_name in all_objectives.items()
        if class_name not in non_core
    ]


def get_objective(objective, return_instance=False, **kwargs):
    """Returns the Objective class corresponding to a given objective name.

    Args:
        objective (str or ObjectiveBase): Name or instance of the objective class.
        return_instance (bool): Whether to return an instance of the objective. This only applies if objective
            is of type str. Note that the instance will be initialized with default arguments.
        kwargs (Any): Any keyword arguments to pass into the objective. Only used when return_instance=True.

    Returns:
        ObjectiveBase if the parameter objective is of type ObjectiveBase. If objective is instead a valid
        objective name, function will return the class corresponding to that name. If return_instance is True,
        an instance of that objective will be returned.

    Raises:
        TypeError: If objective is None.
        TypeError: If objective is not a string and not an instance of ObjectiveBase.
        ObjectiveNotFoundError: If input objective is not a valid objective.
        ObjectiveCreationError: If objective cannot be created properly.
    """
    if objective is None:
        raise TypeError("Objective parameter cannot be NoneType")
    if isinstance(objective, ObjectiveBase):
        return objective
    all_objectives_dict = _all_objectives_dict()
    if not isinstance(objective, str):
        raise TypeError(
            "If parameter objective is not a string, it must be an instance of ObjectiveBase!",
        )
    if objective.lower() not in all_objectives_dict:
        raise ObjectiveNotFoundError(
            f"{objective} is not a valid Objective! "
            "Use evalml.objectives.get_all_objective_names() "
            "to get a list of all valid objective names. ",
        )

    objective_class = all_objectives_dict[objective.lower()]

    if return_instance:
        try:
            return objective_class(**kwargs)
        except TypeError as e:
            raise ObjectiveCreationError(
                f"In get_objective, cannot pass in return_instance=True for {objective} because {str(e)}",
            )

    return objective_class


def get_core_objectives(problem_type):
    """Returns all core objective instances associated with the given problem type.

    Core objectives are designed to work out-of-the-box for any dataset.

    Args:
        problem_type (str/ProblemTypes): Type of problem

    Returns:
        List of ObjectiveBase instances

    Examples:
        >>> for objective in get_core_objectives("regression"):
        ...     print(objective.name)
        ExpVariance
        MaxError
        MedianAE
        MSE
        MAE
        R2
        Root Mean Squared Error
        >>> for objective in get_core_objectives("binary"):
        ...     print(objective.name)
        MCC Binary
        Log Loss Binary
        Gini
        AUC
        Precision
        F1
        Balanced Accuracy Binary
        Accuracy Binary
    """
    problem_type = handle_problem_types(problem_type)
    all_objectives_dict = _all_objectives_dict()
    objectives = [
        obj()
        for obj in all_objectives_dict.values()
        if obj.is_defined_for_problem_type(problem_type)
        and obj not in get_non_core_objectives()
    ]
    return objectives


def get_default_recommendation_objectives(problem_type, imbalanced=False):
    """Get the default recommendation score metrics for the given problem type.

    Args:
        problem_type (str/ProblemType): Type of problem
        imbalanced (boolean): For multiclass problems, if the classes are imbalanced. Defaults to False

    Returns:
        Set of string objective names that correspond to ObjectiveBase objectives
    """
    problem_type = handle_problem_types(problem_type)
    objective_list = [
        obj.name
        for obj in DEFAULT_RECOMMENDATION_OBJECTIVES
        if obj.is_defined_for_problem_type(problem_type)
    ]

    if problem_type == ProblemTypes.MULTICLASS and imbalanced:
        objective_list.remove(objectives.AUCMicro.name)
        objective_list.append(objectives.AUCWeighted.name)
    if problem_type == ProblemTypes.REGRESSION:
        objective_list.remove(objectives.MedianAE.name)
    if problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        objective_list.remove(objectives.R2.name)

    return set(objective_list)


def organize_objectives(problem_type, include=None, exclude=None, imbalanced=False):
    """Generate objectives to consider, with optional modifications to the defaults.

    Args:
        problem_type (str/ProblemType): Type of problem
        include (list[str/ObjectiveBase]): A list of objectives to include beyond the defaults. Defaults to None.
        exclude (list[str/ObjectiveBase]): A list of objectives to exclude from the defaults. Defaults to None.
        imbalanced (boolean): For multiclass problems, if the classes are imbalanced. Defaults to False

    Returns:
        List of string objective names that correspond to ObjectiveBase objectives

    Raises:
        ValueError: If any objectives to include or exclude are not valid for the problem type
        ValueError: If an objective to exclude is not in the default objectives
    """
    problem_type = handle_problem_types(problem_type)
    default_objectives = get_default_recommendation_objectives(problem_type, imbalanced)

    include_objectives = []
    exclude_objectives = []
    if include is not None:
        for objective in include:
            inc_obj = get_objective(objective)
            if not inc_obj.is_defined_for_problem_type(problem_type):
                raise ValueError(
                    f"Objective to include {inc_obj} is not defined for {problem_type}",
                )
            include_objectives.append(inc_obj.name)
    if exclude is not None:
        for objective in exclude:
            ex_obj = get_objective(objective)
            if not ex_obj.is_defined_for_problem_type(problem_type):
                raise ValueError(
                    f"Objective to exclude {ex_obj} is not defined for {problem_type}",
                )
            if ex_obj.name not in default_objectives:
                raise ValueError(
                    f"Cannot exclude objective {ex_obj} because it is not in the default objectives",
                )
            exclude_objectives.append(ex_obj.name)

    default_objectives.update(set(include_objectives))
    return default_objectives - set(exclude_objectives)


def normalize_objectives(objectives_to_normalize, max_objectives, min_objectives):
    """Converts objectives from a [0, inf) scale to [0, 1] given a max and min for each objective.

    Args:
        objectives_to_normalize (dict[str,float]): A dictionary mapping objectives to values
        max_objectives (dict[str,float]): The mapping of objectives to the maximum values for normalization
        min_objectives (dict[str,float]): The mapping of objectives to the minimum values for normalization

    Returns:
        A dictionary mapping objective names to their new normalized values
    """
    normalized = {}
    for objective_name, val in objectives_to_normalize.items():
        objective_obj = get_objective(objective_name)
        # Only normalize objectives that are not bounded like percentages
        # R2 also does not get normalized as it's essentially bounded like a percentage,
        # and we want to penalize aggressively when R2 is negative
        if objective_obj.is_bounded_like_percentage or objective_obj.name == "R2":
            normalized[objective_name] = val
            continue
        max_val, min_val = (
            max_objectives[objective_name],
            min_objectives[objective_name],
        )
        if max_val == min_val:
            normal = 1
        else:
            normal = (val - min_val) / (max_val - min_val)
            if not objective_obj.greater_is_better:
                normal = 1 - normal
        normalized[objective_name] = normal
    return normalized


def recommendation_score(
    objectives,
    prioritized_objective=None,
    custom_weights=None,
):
    """Computes a recommendation score for a model given scores for a group of objectives.

    This recommendation score is a weighted average of the given objectives, by default all weighted equally. Passing
    in a prioritized objective will weight that objective with the prioritized weight, and all other objectives will
    split the remaining weight equally.

    Args:
        objectives (dict[str,float]): A dictionary mapping objectives to their values. Objectives should be a float between
            0 and 1, where higher is better. If the objective does not represent score this way, scores should first be
            normalized using the normalize_objectives function.
        prioritized_objective (str): An optional name of a priority objective that should be given heavier weight (50% of the
            total) than the other objectives contributing to the score. Defaults to None, where all objectives are
            weighted equally.
        custom_weights (dict[str,float]): A dictionary mapping objective names to corresponding weights between 0 and 1.
            If all objectives are listed, should add up to 1. If a subset of objectives are listed, should add up to less
            than 1, and remaining weight will be evenly distributed between the remaining objectives. Should not be used
            at the same time as prioritized_objective.

    Returns:
        A value between 0 and 100 representing how strongly we recommend a pipeline given a set of evaluated objectives

    Raises:
        ValueError: If the objective(s) to prioritize are not in the known objectives, or if the custom weight(s) are not
            a float between 0 and 1.
    """
    objectives = objectives.copy()  # Prevent mutation issues

    if prioritized_objective is not None and custom_weights is not None:
        raise ValueError(
            "Cannot set both prioritized_objective and custom_weights in recommendation score",
        )

    priority_weight = 0
    default_weight = 1 / len(objectives)
    if prioritized_objective is not None:
        if prioritized_objective not in objectives:
            raise ValueError(
                f"Prioritized objective {prioritized_objective} is not in the list of objectives, valid ones are {objectives.keys()}",
            )
        custom_weights = {prioritized_objective: 0.5}

    if custom_weights is not None:
        for objective, objective_weight in custom_weights.items():
            if objective not in objectives:
                raise ValueError(
                    f"Custom weighted objective {objective} does not have a corresponding score",
                )
            if objective_weight <= 0 or objective_weight >= 1:
                raise ValueError(
                    f"Custom weight {objective_weight} for {objective} is not a valid float between 0 and 1",
                )
            objective_val = objectives.pop(objective)
            priority_weight += objective_weight * objective_val
        default_weight = 0
        if len(objectives) > 0:
            remaining_weight = 1 - sum(custom_weights.values())
            default_weight = remaining_weight / len(objectives)

    score_list = [
        objective_value * default_weight for objective_value in objectives.values()
    ]
    score_sum = sum(score_list) + priority_weight
    return 100 * score_sum
