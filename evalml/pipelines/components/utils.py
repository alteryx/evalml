# flake8:noqa
import copy
import inspect
import sys

from .component_base import ComponentBase
from .estimators import (
    CatBoostClassifier,
    CatBoostRegressor,
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
    StandardScaler
)

from evalml.utils import import_or_raise

# When adding new components please import above.
# components_dict() automatically generates dict of components without required parameters


def _components_dict():
    """components_dict() looks through all imported modules and returns all components
        that only have default parameters and can be instantiated"""

    components = dict()
    for _, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        params = inspect.getargspec(obj.__init__)
        if issubclass(obj, ComponentBase):
            if params.defaults:
                if len(params.args) - 1 == len(params.defaults):
                    components[obj.name] = obj
            elif len(params.args) == 1:
                components[obj.name] = obj
    return components


_ALL_COMPONENTS = _components_dict()


def all_components():
    """Returns a complete dict of all supported component classes.

    Returns:
        dict: a dict mapping component name to component class
    """
    components = copy.copy(_ALL_COMPONENTS)
    try:
        import_or_raise("xgboost", error_msg="XGBoost not installed.")
    except ImportError:
        components.pop(XGBoostClassifier.name)
    try:
        import_or_raise("catboost", error_msg="Catboost not installed.")
    except ImportError:
        components.pop(CatBoostClassifier.name)
        components.pop(CatBoostRegressor.name)
    return components
