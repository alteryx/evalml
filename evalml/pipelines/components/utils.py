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
    SimpleImputer,
    StandardScaler)


def components_dict():
    COMPONENTS = dict()
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        print(obj)
        if name not in ['Estimator', 'Transformer']:
            if inspect.isclass(obj):
                COMPONENTS[obj().name] = obj()
    return COMPONENTS

def handle_component(component_str):
    pass