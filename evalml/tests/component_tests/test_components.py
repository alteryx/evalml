import pytest

from evalml.pipelines.components import ComponentBase, Estimator, Transformer


@pytest.fixture
def test_classes():
    class MockComponent(ComponentBase):
        name = "Mock Component"
        _needs_fitting = True
    class MockEstimator(Estimator):
        name = "Mock Estimator"
        _needs_fitting = True
    class MockTransformer(Transformer):
        name = "Mock Transformer"
        _needs_fitting = False
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


def test_missing_attributes(X_y):
    class MockComponentFitting(ComponentBase):
        name = "mock"

    class MockComponentName(ComponentBase):
        _needs_fitting = True

    with pytest.raises(AttributeError, match="Component missing attribute: `name`"):
        MockComponentName(parameters={}, component_obj=None, random_state=0)

    with pytest.raises(AttributeError, match="Component missing attribute: `_needs_fitting`"):
        MockComponentFitting(parameters={}, component_obj=None, random_state=0)


def test_missing_methods_on_components(X_y, test_classes):
    X, y = X_y
    MockComponent, MockEstimator, MockTransformer = test_classes

    component = MockComponent(parameters={}, component_obj=None, random_state=0)
    with pytest.raises(RuntimeError, match="Component requires a fit method or a component_obj that implements fit"):
        component.fit(X)

    estimator = MockEstimator(parameters={}, component_obj=None, random_state=0)
    with pytest.raises(RuntimeError, match="Estimator requires a predict method or a component_obj that implements predict"):
        estimator.predict(X)
    with pytest.raises(RuntimeError, match="Estimator requires a predict_proba method or a component_obj that implements predict_proba"):
        estimator.predict_proba(X)

    transformer = MockTransformer(parameters={}, component_obj=None, random_state=0)
    with pytest.raises(RuntimeError, match="Component requires a fit method or a component_obj that implements fit"):
        transformer.fit(X, y)
    with pytest.raises(RuntimeError, match="Transformer requires a transform method or a component_obj that implements transform"):
        transformer.transform(X)
    with pytest.raises(RuntimeError, match="Transformer requires a fit_transform method or a component_obj that implements fit_transform"):
        transformer.fit_transform(X)


def test_component_fit(X_y):
    X, y = X_y

    class MockEstimator():
        def fit(self, X, y):
            pass

        def predict(self, X):
            pass

    class MockComponent(Estimator):
        name = 'Mock Estimator'
        _needs_fitting = True
        hyperparameter_ranges = {}

        def __init__(self):
            parameters = {}
            est = MockEstimator()
            super().__init__(parameters=parameters,
                             component_obj=est,
                             random_state=0)

    est = MockComponent()
    assert isinstance(est.fit(X, y), ComponentBase)
