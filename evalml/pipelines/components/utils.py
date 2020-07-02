# flake8:noqa
import inspect

from evalml.exceptions import MissingComponentError
from evalml.pipelines import Estimator, Transformer
from evalml.pipelines.components import ComponentBase
from evalml.utils import get_logger
from evalml.utils.gen_utils import get_importable_subclasses

logger = get_logger(__file__)

_all_estimators = get_importable_subclasses(Estimator, args=[], used_in_automl=False)
_all_estimators_used_in_search = get_importable_subclasses(Estimator, args=[], used_in_automl=True)
_all_transformers = get_importable_subclasses(Transformer, args=[], used_in_automl=False)
all_components = _all_estimators + _all_transformers


def handle_component_class(component_class):
    """Standardizes input from a string name to a ComponentBase subclass if necessary.

    If a str is provided, will attempt to look up a ComponentBase class by that name and
    return a new instance. Otherwise if a ComponentBase subclass is provided, will return that
    without modification.

    Arguments:
        component (str, ComponentBase) : input to be standardized

    Returns:
        ComponentBase
    """
    if inspect.isclass(component_class) and issubclass(component_class, ComponentBase):
        return component_class
    if not isinstance(component_class, str):
        raise ValueError(("component_graph may only contain str or ComponentBase subclasses, not '{}'")
                         .format(type(component_class)))
    component_classes = {component.name: component for component in all_components}
    if component_class not in component_classes:
        raise MissingComponentError('Component "{}" was not found'.format(component_class))
    component_class = component_classes[component_class]
    return component_class
