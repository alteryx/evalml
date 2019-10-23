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


def handle_component(component_str):
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
