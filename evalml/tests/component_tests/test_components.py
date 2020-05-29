import numpy as np
import pytest

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    ComponentBase,
    Estimator,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
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


@pytest.fixture
def test_classes():
    class MockComponent(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE

    class MockEstimator(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ['binary']

    class MockTransformer(Transformer):
        name = "Mock Transformer"

    return MockComponent, MockEstimator, MockTransformer


def test_init(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    assert MockComponent({}, None, 0).name == "Mock Component"
    assert MockEstimator({}, None, 0).name == "Mock Estimator"
    assert MockTransformer({}, None, 0).name == "Mock Transformer"


def test_describe(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    params = {'param_a': 'value_a', 'param_b': 123}
    component = MockComponent(params, None, random_state=0)
    assert component.describe(return_dict=True) == {'name': 'Mock Component', 'parameters': params}
    estimator = MockEstimator(params, None, random_state=0)
    assert estimator.describe(return_dict=True) == {'name': 'Mock Estimator', 'parameters': params}
    transformer = MockTransformer(params, None, random_state=0)
    assert transformer.describe(return_dict=True) == {'name': 'Mock Transformer', 'parameters': params}


def test_describe_component():
    enc = OneHotEncoder()
    imputer = SimpleImputer("mean")
    scaler = StandardScaler()
    feature_selection = RFClassifierSelectFromModel(n_estimators=10, number_features=5, percent_features=0.3, threshold=-np.inf)
    assert enc.describe(return_dict=True) == {'name': 'One Hot Encoder', 'parameters': {'top_n': 10}}
    assert imputer.describe(return_dict=True) == {'name': 'Simple Imputer', 'parameters': {'impute_strategy': 'mean', 'fill_value': None}}
    assert scaler.describe(return_dict=True) == {'name': 'Standard Scaler', 'parameters': {}}
    assert feature_selection.describe(return_dict=True) == {'name': 'RF Classifier Select From Model', 'parameters': {'percent_features': 0.3, 'threshold': -np.inf, 'number_features': 5}}

    # testing estimators
    lr_classifier = LogisticRegressionClassifier()
    et_classifier = ExtraTreesClassifier(n_estimators=10, max_features="auto")
    et_regressor = ExtraTreesRegressor(n_estimators=10, max_features="auto")
    rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=3)
    rf_regressor = RandomForestRegressor(n_estimators=10, max_depth=3)
    linear_regressor = LinearRegressor()
    assert lr_classifier.describe(return_dict=True) == {'name': 'Logistic Regression Classifier', 'parameters': {'C': 1.0, 'penalty': 'l2'}}
    assert et_classifier.describe(return_dict=True) == {'name': 'Extra Trees Classifier', 'parameters': {'max_depth': 6, 'max_features': "auto", 'n_estimators': 10}}
    assert et_regressor.describe(return_dict=True) == {'name': 'Extra Trees Regressor', 'parameters': {'max_depth': 6, 'max_features': "auto", 'n_estimators': 10}}
    assert rf_classifier.describe(return_dict=True) == {'name': 'Random Forest Classifier', 'parameters': {'max_depth': 3, 'n_estimators': 10}}
    assert rf_regressor.describe(return_dict=True) == {'name': 'Random Forest Regressor', 'parameters': {'max_depth': 3, 'n_estimators': 10}}
    assert linear_regressor.describe(return_dict=True) == {'name': 'Linear Regressor', 'parameters': {'fit_intercept': True, 'normalize': False}}
    try:
        xgb_classifier = XGBoostClassifier(eta=0.1, min_child_weight=1, max_depth=3, n_estimators=75)
        assert xgb_classifier.describe(return_dict=True) == {'name': 'XGBoost Classifier', 'parameters': {'eta': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 75}}
    except ImportError:
        pass


def test_missing_attributes(X_y):
    class MockComponentName(ComponentBase):
        model_family = ModelFamily.NONE

    with pytest.raises(TypeError):
        MockComponentName(parameters={}, component_obj=None, random_state=0)

    class MockComponentModelFamily(ComponentBase):
        name = "Mock Component"

    with pytest.raises(TypeError):
        MockComponentModelFamily(parameters={}, component_obj=None, random_state=0)

    class MockEstimator(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL

    with pytest.raises(TypeError):
        MockEstimator(parameters={}, component_obj=None, random_state=0)


def test_missing_methods_on_components(X_y, test_classes):
    X, y = X_y
    MockComponent, MockEstimator, MockTransformer = test_classes

    class MockTransformerWithFit(Transformer):
        name = "Mock Transformer"

        def fit(self, X, y=None):
            return X

    component = MockComponent(parameters={}, component_obj=None, random_state=0)
    with pytest.raises(MethodPropertyNotFoundError, match="Component requires a fit method or a component_obj that implements fit"):
        component.fit(X)

    estimator = MockEstimator(parameters={}, component_obj=None, random_state=0)
    with pytest.raises(MethodPropertyNotFoundError, match="Estimator requires a predict method or a component_obj that implements predict"):
        estimator.predict(X)
    with pytest.raises(MethodPropertyNotFoundError, match="Estimator requires a predict_proba method or a component_obj that implements predict_proba"):
        estimator.predict_proba(X)

    transformer = MockTransformer(parameters={}, component_obj=None, random_state=0)
    transformer_with_fit = MockTransformerWithFit(parameters={}, component_obj=None, random_state=0)
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
        def fit(self, X, y):
            pass

    class MockComponent(Estimator):
        name = 'Mock Estimator'
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ['binary']
        hyperparameter_ranges = {}

        def __init__(self):
            parameters = {}
            est = MockEstimator()
            super().__init__(parameters=parameters,
                             component_obj=est,
                             random_state=0)

    est = MockComponent()
    assert isinstance(est.fit(X, y), ComponentBase)


def test_component_fit_transform(X_y):
    X, y = X_y

    class MockTransformerWithFitTransform(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit_transform(self, X, y=None):
            return X

        def __init__(self):
            parameters = {}
            super().__init__(parameters=parameters,
                             component_obj=None,
                             random_state=0)

    class MockTransformerWithFitTransformButError(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit_transform(self, X, y=None):
            raise RuntimeError

        def __init__(self):
            parameters = {}
            super().__init__(parameters=parameters,
                             component_obj=None,
                             random_state=0)

    class MockTransformerWithFitAndTransform(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit(self, X, y=None):
            return X

        def transform(self, X, y=None):
            return X

        def __init__(self):
            parameters = {}
            super().__init__(parameters=parameters,
                             component_obj=None,
                             random_state=0)

    class MockTransformerWithOnlyFit(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit(self, X, y=None):
            return X

        def __init__(self):
            parameters = {}
            super().__init__(parameters=parameters,
                             component_obj=None,
                             random_state=0)

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


def test_regressor_call_predict_proba(test_classes):
    X = np.array([])
    _, MockEstimator, _ = test_classes
    component = MockEstimator({}, None, 0)
    with pytest.raises(MethodPropertyNotFoundError):
        component.predict_proba(X)


def test_component_describe(test_classes, caplog):
    MockComponent, _, _ = test_classes
    component = MockComponent({}, None, 0)
    component.describe(print_name=True)
    out = caplog.text
    assert "Mock Component" in out
