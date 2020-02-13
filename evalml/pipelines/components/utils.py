# flake8:noqa
import inspect
import sys

from .component_base import ComponentBase
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


def handle_component(component):
    """Handles component by either returning the ComponentBase object or converts the str

        Args:
            component_type (str or ComponentBase) : component that needs to be handled

        Returns:
            ComponentBase
    """
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
