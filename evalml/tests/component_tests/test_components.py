import copy
import numpy as np
import pytest

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    ComponentBase,
    Estimator,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    RFClassifierSelectFromModel,
    SimpleImputer,
    StandardScaler,
    Transformer,
    XGBoostClassifier
)
from evalml.utils import get_random_state
from evalml.pipelines.components.validation_error import ValidationError


@pytest.fixture
def test_classes():
    class MockComponent(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        def __init__(self, param_a=1, param_b='two', random_state=0):
            self.param_a = param_a
            self.param_b = param_b
            super().__init__(random_state=random_state)

    class MockEstimator(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ['binary']
        def __init__(self, param_a=1, param_b='two', random_state=0):
            self.param_a = param_a
            self.param_b = param_b
            super().__init__(random_state=random_state)

    class MockTransformer(Transformer):
        name = "Mock Transformer"
        def __init__(self, param_a=1, param_b='two', random_state=0):
            self.param_a = param_a
            self.param_b = param_b
            super().__init__(random_state=random_state)

    return MockComponent, MockEstimator, MockTransformer


def test_init(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    assert MockComponent()
    assert MockEstimator()
    assert MockTransformer()


def split_random_state(parameters):
    params = copy.copy(parameters)
    random_state = params.pop('random_state')
    return params, random_state


def test_default_parameters(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    assert MockComponent.default_parameters == {'param_a': 1, 'param_b': 'two', 'random_state': 0}

    class MockComponent(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        def __init__(self, param_a=1, param_b=None, random_state=np.random.RandomState(42)):
            self.param_a = param_a
            self.param_b = param_b
            super().__init__(random_state=random_state)

    dp, random_state = split_random_state(MockComponent.default_parameters)
    assert dp == {'param_a': 1, 'param_b': None}
    assert random_state.rand() == np.random.RandomState(42).rand()


def test_parameters(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    c = MockComponent()
    p, random_state = split_random_state(c.parameters)
    assert p == {'param_a': 1, 'param_b': 'two'}
    assert random_state.rand() == np.random.RandomState(0).rand()

    c = MockComponent(param_b='three')
    p, random_state = split_random_state(c.parameters)
    assert p == {'param_a': 1, 'param_b': 'three'}
    assert random_state.rand() == np.random.RandomState(0).rand()

    c = MockComponent(param_b='three', random_state=np.random.RandomState(42))
    p, random_state = split_random_state(c.parameters)
    assert p == {'param_a': 1, 'param_b': 'three'}
    assert random_state.rand() == np.random.RandomState(42).rand()

    c = MockComponent(param_b=None, random_state=np.random.RandomState(42))
    p, random_state = split_random_state(c.parameters)
    assert p == {'param_a': 1, 'param_b': None}
    assert random_state.rand() == np.random.RandomState(42).rand()


def test_init_invalid_args():
    class MockComponentNoInit(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE

    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ should not provide argument 'component_obj'"):
        MockComponentNoInit.default_parameters
    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ should not provide argument 'component_obj'"):
        MockComponentNoInit()

    class MockComponentFixedArg(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        def __init__(self, param_a, param_b='two', random_state=0):
            self.param_a = param_a
            self.param_b = param_b
            super().__init__(random_state=random_state)

    with pytest.raises(TypeError, match=r"__init__\(\) missing 1 required positional argument: 'param_a'"):
        MockComponentFixedArg()
    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ has no default value for argument 'param_a'"):
        MockComponentFixedArg.default_parameters
    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ has no default value for argument 'param_a'"):
        MockComponentFixedArg(1)

    class MockComponentVarArgs(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        def __init__(self, param_a=1, param_b='two', random_state=0, *args):
            self.param_a = param_a
            self.param_b = param_b
            super().__init__(random_state=random_state)

    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ uses \*args or \*\*kwargs"):
        MockComponentVarArgs.default_parameters
    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ uses \*args or \*\*kwargs"):
        MockComponentVarArgs()

    class MockComponentKWArgs(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        def __init__(self, param_a=1, param_b='two', random_state=0, **kwargs):
            self.param_a = param_a
            self.param_b = param_b
            super().__init__(random_state=random_state)

    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ uses \*args or \*\*kwargs"):
        MockComponentKWArgs.default_parameters
    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ uses \*args or \*\*kwargs"):
        MockComponentKWArgs()

    class MockComponentNoSavedState(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        def __init__(self, param_a=1, param_b='two', random_state=0):
            self.param_b = param_b
            super().__init__(random_state=random_state)

    # unfortunately, since our validation for this case can only run at instantiation, default_parameters doesn't throw.
    MockComponentNoSavedState.default_parameters
    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ has not saved state for parameter 'param_a'"):
        MockComponentNoSavedState()

    class MockComponentMissingRandomState(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        def __init__(self, param_a=1, param_b='two'):
            self.param_a = param_a
            self.param_b = param_b
            super().__init__()

    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ missing values for required parameters: '{'random_state'}'"):
        MockComponentMissingRandomState.default_parameters
    with pytest.raises(ValidationError, match=r"Component 'Mock Component' __init__ missing values for required parameters: '{'random_state'}'"):
        MockComponentMissingRandomState()


def test_random_state(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    c = MockComponent(random_state=42)
    assert c.random_state.rand()
    assert (c.random_state.get_state()[0] == np.random.RandomState(42).get_state()[0])
    assert MockEstimator(random_state=0)
    assert MockTransformer(random_state=0)


def test_name(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    assert MockComponent.name == "Mock Component"
    assert MockEstimator.name == "Mock Estimator"
    assert MockTransformer.name == "Mock Transformer"


def split_random_state_description(description):
    descr = copy.deepcopy(description)
    random_state = descr['parameters'].pop('random_state')
    return descr, random_state


def test_describe(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    params = {'param_a': 'value_a', 'param_b': 123}
    component = MockComponent(**params, random_state=0)
    result, _ = split_random_state_description(component.describe(return_dict=True))
    assert result == {'name': 'Mock Component', 'parameters': params}

    component = MockTransformer(**params, random_state=0)
    result, _ = split_random_state_description(component.describe(return_dict=True))
    assert result == {'name': 'Mock Transformer', 'parameters': params}

    component = MockEstimator(**params, random_state=0)
    result, _ = split_random_state_description(component.describe(return_dict=True))
    assert result == {'name': 'Mock Estimator', 'parameters': params}


def test_describe_component():
    enc = OneHotEncoder()
    imputer = SimpleImputer(impute_strategy='mean')
    scaler = StandardScaler()
    feature_selection = RFClassifierSelectFromModel(n_estimators=10, number_features=5, percent_features=0.3, threshold=-np.inf)
    assert split_random_state_description(enc.describe(return_dict=True))[0] == {'name': 'One Hot Encoder', 'parameters': {'top_n': 10}}
    assert split_random_state_description(imputer.describe(return_dict=True))[0] == {'name': 'Simple Imputer', 'parameters': {'impute_strategy': 'mean', 'fill_value': None}}
    assert split_random_state_description(scaler.describe(return_dict=True))[0] == {'name': 'Standard Scaler', 'parameters': {}}
    assert split_random_state_description(feature_selection.describe(return_dict=True))[0] == {'name': 'RF Classifier Select From Model', 'parameters': {'n_estimators': 10, 'number_features': 5, 'percent_features': 0.3, 'threshold': -np.inf, 'n_jobs': -1, 'max_depth': None}}

    # testing estimators
    lr_classifier = LogisticRegressionClassifier()
    rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=3)
    rf_regressor = RandomForestRegressor(n_estimators=10, max_depth=3)
    linear_regressor = LinearRegressor()
    assert split_random_state_description(lr_classifier.describe(return_dict=True))[0] == {'name': 'Logistic Regression Classifier', 'parameters': {'C': 1.0, 'n_jobs': -1, 'penalty': 'l2'}}
    assert split_random_state_description(rf_classifier.describe(return_dict=True))[0] == {'name': 'Random Forest Classifier', 'parameters': {'max_depth': 3, 'n_estimators': 10, 'n_jobs': -1}}
    assert split_random_state_description(rf_regressor.describe(return_dict=True))[0] == {'name': 'Random Forest Regressor', 'parameters': {'max_depth': 3, 'n_estimators': 10, 'n_jobs': -1}}
    assert split_random_state_description(linear_regressor.describe(return_dict=True))[0] == {'name': 'Linear Regressor', 'parameters': {'fit_intercept': True, 'n_jobs': -1, 'normalize': False}}
    try:
        xgb_classifier = XGBoostClassifier(eta=0.1, min_child_weight=1, max_depth=3, n_estimators=75)
        assert split_random_state_description(xgb_classifier.describe(return_dict=True))[0] == {'name': 'XGBoost Classifier', 'parameters': {'eta': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 75}}
    except ImportError:
        pass


def test_missing_attributes(X_y):
    class MockComponentName(ComponentBase):
        model_family = ModelFamily.NONE

    with pytest.raises(TypeError):
        MockComponentName(random_state=0)

    class MockComponentModelFamily(ComponentBase):
        name = "Mock Component"

    with pytest.raises(TypeError):
        MockComponentModelFamily(random_state=0)

    class MockEstimator(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL

    with pytest.raises(TypeError):
        MockEstimator(random_state=0)


def test_missing_methods_on_components(X_y, test_classes):
    X, y = X_y
    MockComponent, MockEstimator, MockTransformer = test_classes

    class MockTransformerWithFit(Transformer):
        name = "Mock Transformer"

        def __init__(self, random_state=0):
            super().__init__(random_state=random_state)

        def fit(self, X, y=None):
            return X

    component = MockComponent(random_state=0)
    with pytest.raises(MethodPropertyNotFoundError, match="Component requires a fit method or a component_obj that implements fit"):
        component.fit(X)

    estimator = MockEstimator(random_state=0)
    with pytest.raises(MethodPropertyNotFoundError, match="Estimator requires a predict method or a component_obj that implements predict"):
        estimator.predict(X)
    with pytest.raises(MethodPropertyNotFoundError, match="Estimator requires a predict_proba method or a component_obj that implements predict_proba"):
        estimator.predict_proba(X)

    transformer = MockTransformer(random_state=0)
    transformer_with_fit = MockTransformerWithFit(random_state=0)
    with pytest.raises(MethodPropertyNotFoundError, match="Component requires a fit method or a component_obj that implements fit"):
        transformer.fit(X, y)
    with pytest.raises(MethodPropertyNotFoundError, match="Transformer requires a transform method or a component_obj that implements transform"):
        transformer.transform(X)
    with pytest.raises(MethodPropertyNotFoundError, match="Component requires a fit method or a component_obj that implements fit"):
        transformer.fit_transform(X)
    with pytest.raises(MethodPropertyNotFoundError, match="Transformer requires a transform method or a component_obj that implements transform"):
        transformer_with_fit.fit_transform(X)


def test_component_fit(X_y):
    X, y = X_y

    class MockEstimator():
        def __init__(self):
            pass

        def fit(self, X, y):
            pass

        def predict(self, X):
            raise NotImplementedError()

    class MockComponent(Estimator):
        name = 'Mock Estimator'
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ['binary']
        hyperparameter_ranges = {}

        def __init__(self, random_state=0):
            est = MockEstimator()
            super().__init__(component_obj=est,
                             random_state=random_state)

    est = MockComponent()
    assert isinstance(est.fit(X, y), ComponentBase)


def test_component_fit_transform(X_y):
    X, y = X_y

    class MockTransformerWithFitTransform(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def __init__(self, random_state=0):
            super().__init__(random_state=random_state)

        def fit_transform(self, X, y=None):
            return X

        def __init__(self, random_state=0):
            super().__init__(random_state=0)

    class MockTransformerWithFitTransformButError(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def __init__(self, random_state=0):
            super().__init__(random_state=random_state)

        def fit_transform(self, X, y=None):
            raise RuntimeError

        def __init__(self, random_state=0):
            super().__init__(random_state=0)

    class MockTransformerWithFitAndTransform(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit(self, X, y=None):
            return X

        def transform(self, X, y=None):
            return X

        def __init__(self, random_state=0):
            super().__init__(random_state=random_state)

    class MockTransformerWithOnlyFit(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit(self, X, y=None):
            return X

        def __init__(self, random_state=0):
            super().__init__(random_state=random_state)

    component = MockTransformerWithFitTransform()
    assert isinstance(component.fit_transform(X, y), np.ndarray)

    component = MockTransformerWithFitTransformButError()
    with pytest.raises(RuntimeError):
        component.fit_transform(X, y)

    component = MockTransformerWithFitAndTransform()
    assert isinstance(component.fit_transform(X, y), np.ndarray)

    component = MockTransformerWithOnlyFit()
    with pytest.raises(MethodPropertyNotFoundError):
        component.fit_transform(X, y)


def test_model_family_components(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes

    assert MockComponent.model_family == ModelFamily.NONE
    assert MockTransformer.model_family == ModelFamily.NONE
    assert MockEstimator.model_family == ModelFamily.LINEAR_MODEL
