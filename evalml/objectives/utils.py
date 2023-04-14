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
            "Use evalml.objectives.get_all_objective_names()"
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


def get_default_objectives(problem_type, imbalanced=False):
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

    return set(objectives)


def organize_objectives(include, exclude, problem_type, imbalanced=False):
    """Given modifications to the default set of objectives, update the list.

    Args:
        include (list[str/ObjectiveBase]): A list of objectives to include beyond the defaults
        exclude (list[str/ObjectiveBase]): A list of objectives to exclude from the defaults
        problem_type (str/ProblemType): Type of problem
        imbalanced (boolean): For multiclass problems, if the classes are imbalanced. Defaults to False

    Returns:
        List of string objective names that correspond to ObjectiveBase objectives
    """
    problem_type = handle_problem_types(problem_type)
    default_objectives = get_default_objectives(problem_type, imbalanced)

    include_objectives = []
    exclude_objectives = []
    for objective in include:
        inc_obj = get_objective(objective)
        if not inc_obj.is_defined_for_problem_type(problem_type):
            raise ValueError(
                f"Objective to include {inc_obj} is not defined for {problem_type}",
            )
        include_objectives.append(inc_obj)
    for objective in exclude:
        ex_obj = get_objective(objective)
        if not ex_obj.is_defined_for_problem_type(problem_type):
            raise ValueError(
                f"Objective to exclude {ex_obj} is not defined for {problem_type}",
            )
        if ex_obj.name not in default_objectives:
            raise ValueError(
                f"Objective to exlude {ex_obj} is not in the default objectives {default_objectives}",
            )
        exclude_objectives.append(ex_obj)

    default_objectives.update(set(include_objectives))
    return default_objectives - set(exclude_objectives)
