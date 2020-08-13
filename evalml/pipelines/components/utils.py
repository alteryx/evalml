# flake8:noqa
import inspect

from evalml.exceptions import MissingComponentError
from evalml.model_family.utils import handle_model_family
from evalml.pipelines.components import ComponentBase, Estimator, Transformer
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_logger
from evalml.utils.gen_utils import get_importable_subclasses

logger = get_logger(__file__)


def _all_estimators():
    return get_importable_subclasses(Estimator, used_in_automl=False)


def _all_estimators_used_in_search():
    return get_importable_subclasses(Estimator, used_in_automl=True)


def _all_transformers():
    return get_importable_subclasses(Transformer, used_in_automl=False)


def all_components():
    return _all_estimators() + _all_transformers()


def allowed_model_families(problem_type):
    """List the model types allowed for a particular problem type.

    Arguments:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        list[ModelFamily]: a list of model families
    """

    estimators = []
    problem_type = handle_problem_types(problem_type)
    for estimator in _all_estimators_used_in_search():
        if problem_type in set(handle_problem_types(problem) for problem in estimator.supported_problem_types):
            estimators.append(estimator)

    return list(set([e.model_family for e in estimators]))


def get_estimators(problem_type, model_families=None):
    """Returns the estimators allowed for a particular problem type.

    Can also optionally filter by a list of model types.

    Arguments:
        problem_type (ProblemTypes or str): problem type to filter for
        model_families (list[ModelFamily] or list[str]): model families to filter for

    Returns:
        list[class]: a list of estimator subclasses
    """
    if model_families is not None and not isinstance(model_families, list):
        raise TypeError("model_families parameter is not a list.")
    problem_type = handle_problem_types(problem_type)
    if model_families is None:
        model_families = allowed_model_families(problem_type)

    model_families = [handle_model_family(model_family) for model_family in model_families]
    all_model_families = allowed_model_families(problem_type)
    for model_family in model_families:
        if model_family not in all_model_families:
            raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_family))

    estimator_classes = []
    for estimator_class in _all_estimators_used_in_search():
        if problem_type not in [handle_problem_types(supported_pt) for supported_pt in estimator_class.supported_problem_types]:
            continue
        if estimator_class.model_family not in model_families:
            continue
        estimator_classes.append(estimator_class)
    return estimator_classes


def handle_component_class(component_class):
    """Standardizes input from a string name to a ComponentBase subclass if necessary.

    If a str is provided, will attempt to look up a ComponentBase class by that name and
    return a new instance. Otherwise if a ComponentBase subclass is provided, will return that
    without modification.

    Arguments:
        component (str, ComponentBase): input to be standardized

    Returns:
        ComponentBase
    """
    if inspect.isclass(component_class) and issubclass(component_class, ComponentBase):
        return component_class
    if not isinstance(component_class, str):
        raise ValueError(("component_graph may only contain str or ComponentBase subclasses, not '{}'")
                         .format(type(component_class)))
    component_classes = {component.name: component for component in all_components()}
    if component_class not in component_classes:
        raise MissingComponentError('Component "{}" was not found'.format(component_class))
    component_class = component_classes[component_class]
    return component_class
