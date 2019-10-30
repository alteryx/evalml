# flake8:noqa
import inspect
import sys

from .component_base import ComponentBase
from .component_types import ComponentTypes
from .estimators import (
    Estimator,
    LinearRegressor,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBoostClassifier
)
from .transformers import (
    CategoricalEncoder,
    FeatureSelector,
    OneHotEncoder,
    RFClassifierSelectFromModel,
    RFRegressorSelectFromModel,
    SimpleImputer,
    StandardScaler,
    Transformer
)

# When adding new components please import above.
# components_dict() automatically generates dict of components without required parameters


def __components_dict():
    """components_dict() looks through all imported modules and returns all components
        that only have default parameters and can be instantiated"""

    COMPONENTS = dict()
    for _, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        params = inspect.getargspec(obj.__init__)
        if issubclass(obj, ComponentBase):
            if params.defaults:
                if len(params.args) - 1 == len(params.defaults):
                    COMPONENTS[obj.name] = obj
            elif len(params.args) == 1:
                COMPONENTS[obj.name] = obj
    return COMPONENTS

__COMPONENTS = __components_dict()

DEFAULT_COMPONENTS = {
   ComponentTypes.CLASSIFIER: LogisticRegressionClassifier,
   ComponentTypes.CATEGORICAL_ENCODER: OneHotEncoder,
   ComponentTypes.IMPUTER: SimpleImputer,
   ComponentTypes.REGRESSOR: LinearRegressor,
   ComponentTypes.SCALER: StandardScaler,
   ComponentTypes.FEATURE_SELECTION: RFClassifierSelectFromModel
}


def handle_component(component):
    """Handles component by either returning the ComponentBase object or converts the str

        Args:
            component_type (str or ComponentBase) : component that needs to be handled

        Returns:
            ComponentBase
    """
    try:
        component_type = str_to_component_type(component)
        component_class = DEFAULT_COMPONENTS[component_type]
        component = component_class()
    except ValueError:
        component = str_to_component(component)
    return component


def str_to_component(component):
    try:
        if isinstance(component, str):
            component_class = __COMPONENTS[component]
            return component_class()
        elif isinstance(component, ComponentBase):
            return component
        else:
            raise ValueError("handle_component only takes in str or ComponentBase")
    except KeyError:
        raise ValueError("Component {} has required parameters and string initialization is not supported".format(component))


def str_to_component_type(component_type):
    """Handles component_type by either returning the ComponentTypes object or converts the str

        Args:
            component_type (str or ComponentTypes) : component type that needs to be handled

        Returns:
            ComponentTypes
    """
    if isinstance(component_type, str):
        try:
            tpe = ComponentTypes[component_type.upper()]
        except KeyError:
            raise ValueError('Component type \'{}\' does not exist'.format(component_type))
        return tpe
    if isinstance(component_type, ComponentTypes):
        return component_type
    raise ValueError('`handle_component_types` was not passed a str or ComponentTypes object')
