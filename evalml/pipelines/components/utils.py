# flake8:noqa
import inspect

from evalml.exceptions import MissingComponentError
from evalml.pipelines.components import ComponentBase, Estimator, Transformer
from evalml.utils import get_logger
from evalml.utils.gen_utils import get_importable_subclasses

logger = get_logger(__file__)

# We need to differentiate between all estimators, and those we use for search.
# We use the former for unit tests and the latter for creating pipelines.
_estimator_message = 'Estimator {} failed import, withholding from all_estimators'
_all_estimators = get_importable_subclasses(Estimator, args=[],
                                            message=_estimator_message,
                                            used_in_automl=False)
all_estimators_used_in_search = get_importable_subclasses(Estimator, args=[],
                                                          message=_estimator_message,
                                                          used_in_automl=True)

all_transformers = get_importable_subclasses(Transformer, args=[],
                                             message='Transformer {} failed import, withholding from all_transformers',
                                             used_in_automl=False)


def all_components():
    """Return all components (even if they are not used for automl search).

    Returns:
        List of all component classes.
    """
    return _all_estimators() + all_transformers()


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
    component_classes = {component.name: component for component in all_components()}
    if component_class not in component_classes:
        raise MissingComponentError('Component "{}" was not found'.format(component_class))
    component_class = component_classes[component_class]
    return component_class
