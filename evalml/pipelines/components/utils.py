# flake8:noqa
import copy
import inspect
import sys

from .component_base import ComponentBase
from .estimators import (
    BaselineClassifier,
    BaselineRegressor,
    CatBoostClassifier,
    CatBoostRegressor,
    ElasticNetClassifier,
    ElasticNetRegressor,
    Estimator,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    LinearRegressor,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBoostClassifier,
    XGBoostRegressor
)
from .transformers import (
    CategoricalEncoder,
    DropColumnsTransformer,
    FeatureSelector,
    OneHotEncoder,
    RFClassifierSelectFromModel,
    RFRegressorSelectFromModel,
    SimpleImputer,
    StandardScaler,
    Transformer
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
        components.pop(XGBoostRegressor.name)
    try:
        import_or_raise("catboost", error_msg="Catboost not installed.")
    except ImportError:
        components.pop(CatBoostClassifier.name)
        components.pop(CatBoostRegressor.name)
    return components


def handle_component(component):
    """Standardizes input to a new ComponentBase instance if necessary.

    If a str is provided, will attempt to look up a ComponentBase class by that name and
    return a new instance. Otherwise if a ComponentBase instance is provided, will return that
    without modification.

    Arguments:
        component (str, ComponentBase) : input to be standardized

    Returns:
        ComponentBase
    """
    if isinstance(component, ComponentBase):
        return component
    if not isinstance(component, str):
        raise ValueError("handle_component only takes in str or ComponentBase")
    components = all_components()
    if component not in components:
        raise KeyError("Component {} was not found".format(component))
    component_class = all_components()[component]
    return component_class()
