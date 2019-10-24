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
    Encoder,
    FeatureSelector,
    OneHotEncoder,
    RFSelectFromModel,
    SelectFromModel,
    SimpleImputer,
    StandardScaler,
    Transformer
)


def components_dict():
    COMPONENTS = dict()
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            try:
                COMPONENTS[obj().name] = obj()
            except Exception:
                pass
    return COMPONENTS

COMPONENTS = components_dict()

DEFAULT_COMPONENTS = {
   ComponentTypes.CLASSIFIER: LogisticRegressionClassifier,
   ComponentTypes.ENCODER: OneHotEncoder,
   ComponentTypes.IMPUTER: SimpleImputer,
   ComponentTypes.REGRESSOR: LinearRegressor,
   ComponentTypes.SCALER: StandardScaler,
   ComponentTypes.FEATURE_SELECTION: RFSelectFromModel
}


def handle_component(component_str):
    """Handles component_str by either returning the ComponentBase object or converts the str

        Args:
            component_type (str or ComponentBase) : component that needs to be handled

        Returns:
            ComponentBase
    """
    try:
        component_type = handle_component_types(component_str)
        component_class = DEFAULT_COMPONENTS[component_type]
        component = component_class()
    except ValueError:
        try:
            if isinstance(component_str, str):
                component = COMPONENTS[component_str]
            elif isinstance(component_str, ComponentBase):
                return component_str
            else:
                raise ValueError("handle_component only takes in str or ComponentBase")
        except KeyError:
            raise ValueError("Component {} has required parameters and string initialization is not supproted".format(component_str))
    return component


def handle_component_types(component_type):
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
