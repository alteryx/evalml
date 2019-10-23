import sys, inspect

from .estimators import (
    Estimator,
    LinearRegressor,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBoostClassifier
)
from .transformers import (
    Transformer,
    OneHotEncoder,
    SelectFromModel,
    RFSelectFromModel,
    SimpleImputer,
    StandardScaler,
    FeatureSelector,
    Encoder
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
        component = COMPONENTS[component_str]
    except KeyError:
        raise ValueError("Component {} has required parameters and string initialization is not supproted".format(component_str))
    return component