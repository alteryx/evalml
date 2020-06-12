import inspect

import numpy as np
import pytest

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    ComponentBase,
    DropColumns,
    ElasticNetClassifier,
    ElasticNetRegressor,
    Estimator,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    PerColumnImputer,
    RandomForestClassifier,
    RandomForestRegressor,
    RFClassifierSelectFromModel,
    SimpleImputer,
    StandardScaler,
    Transformer,
    XGBoostClassifier,
    all_components
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
    assert MockComponent().name == "Mock Component"
    assert MockEstimator().name == "Mock Estimator"
    assert MockTransformer().name == "Mock Transformer"


def test_describe(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    params = {'param_a': 'value_a', 'param_b': 123}
    component = MockComponent(parameters=params)
    assert component.describe(return_dict=True) == {'name': 'Mock Component', 'parameters': params}
    estimator = MockEstimator(parameters=params)
    assert estimator.describe(return_dict=True) == {'name': 'Mock Estimator', 'parameters': params}
    transformer = MockTransformer(parameters=params)
    assert transformer.describe(return_dict=True) == {'name': 'Mock Transformer', 'parameters': params}


def test_describe_component():
    enc = OneHotEncoder()
    imputer = SimpleImputer("mean")
    column_imputer = PerColumnImputer({"a": "mean", "b": ("constant", 100)})
    scaler = StandardScaler()
    feature_selection = RFClassifierSelectFromModel(n_estimators=10, number_features=5, percent_features=0.3, threshold=-np.inf)
    drop_col_transformer = DropColumns(columns=['col_one', 'col_two'])
    assert enc.describe(return_dict=True) == {'name': 'One Hot Encoder', 'parameters': {'top_n': 10}}
    assert imputer.describe(return_dict=True) == {'name': 'Simple Imputer', 'parameters': {'impute_strategy': 'mean', 'fill_value': None}}
    assert column_imputer.describe(return_dict=True) == {'name': 'Per Column Imputer', 'parameters': {'impute_strategies': {'a': 'mean', 'b': ('constant', 100)}, 'default_impute_strategy': 'most_frequent'}}
    assert scaler.describe(return_dict=True) == {'name': 'Standard Scaler', 'parameters': {}}
    assert feature_selection.describe(return_dict=True) == {'name': 'RF Classifier Select From Model', 'parameters': {'number_features': 5, 'n_estimators': 10, 'max_depth': None, 'percent_features': 0.3, 'threshold': -np.inf, 'n_jobs': -1}}
    assert drop_col_transformer.describe(return_dict=True) == {'name': 'Drop Columns Transformer', 'parameters': {'columns': ['col_one', 'col_two']}}

    # testing estimators
    lr_classifier = LogisticRegressionClassifier()
    en_classifier = ElasticNetClassifier()
    en_regressor = ElasticNetRegressor()
    et_classifier = ExtraTreesClassifier(n_estimators=10, max_features="auto")
    et_regressor = ExtraTreesRegressor(n_estimators=10, max_features="auto")
    rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=3)
    rf_regressor = RandomForestRegressor(n_estimators=10, max_depth=3)
    linear_regressor = LinearRegressor()
    assert lr_classifier.describe(return_dict=True) == {'name': 'Logistic Regression Classifier', 'parameters': {'penalty': 'l2', 'C': 1.0, 'n_jobs': -1}}
    assert en_classifier.describe(return_dict=True) == {'name': 'Elastic Net Classifier', 'parameters': {'alpha': 0.5, 'l1_ratio': 0.5, 'n_jobs': -1, 'max_iter': 1000}}
    assert en_regressor.describe(return_dict=True) == {'name': 'Elastic Net Regressor', 'parameters': {'alpha': 0.5, 'l1_ratio': 0.5, 'max_iter': 1000, 'normalize': False}}
    assert et_classifier.describe(return_dict=True) == {'name': 'Extra Trees Classifier', 'parameters': {'n_estimators': 10, 'max_features': 'auto', 'max_depth': 6, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_jobs': -1}}
    assert et_regressor.describe(return_dict=True) == {'name': 'Extra Trees Regressor', 'parameters': {'n_estimators': 10, 'max_features': 'auto', 'max_depth': 6, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_jobs': -1}}
    assert rf_classifier.describe(return_dict=True) == {'name': 'Random Forest Classifier', 'parameters': {'n_estimators': 10, 'max_depth': 3, 'n_jobs': -1}}
    assert rf_regressor.describe(return_dict=True) == {'name': 'Random Forest Regressor', 'parameters': {'n_estimators': 10, 'max_depth': 3, 'n_jobs': -1}}
    assert linear_regressor.describe(return_dict=True) == {'name': 'Linear Regressor', 'parameters': {'fit_intercept': True, 'normalize': False, 'n_jobs': -1}}
    try:
        xgb_classifier = XGBoostClassifier(eta=0.1, min_child_weight=1, max_depth=3, n_estimators=75)
        assert xgb_classifier.describe(return_dict=True) == {'name': 'XGBoost Classifier', 'parameters': {'eta': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 75}}
    except ImportError:
        pass


def test_missing_attributes(X_y):
    class MockComponentName(ComponentBase):
        model_family = ModelFamily.NONE

    with pytest.raises(TypeError):
        MockComponentName()

    class MockComponentModelFamily(ComponentBase):
        name = "Mock Component"

    with pytest.raises(TypeError):
        MockComponentModelFamily()

    class MockEstimator(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL

    with pytest.raises(TypeError):
        MockEstimator()


def test_missing_methods_on_components(X_y, test_classes):
    X, y = X_y
    MockComponent, MockEstimator, MockTransformer = test_classes

    class MockTransformerWithFit(Transformer):
        name = "Mock Transformer"

        def fit(self, X, y=None):
            return X

    component = MockComponent()
    with pytest.raises(MethodPropertyNotFoundError, match="Component requires a fit method or a component_obj that implements fit"):
        component.fit(X)

    estimator = MockEstimator()
    with pytest.raises(MethodPropertyNotFoundError, match="Estimator requires a predict method or a component_obj that implements predict"):
        estimator.predict(X)
    with pytest.raises(MethodPropertyNotFoundError, match="Estimator requires a predict_proba method or a component_obj that implements predict_proba"):
        estimator.predict_proba(X)

    transformer = MockTransformer()
    transformer_with_fit = MockTransformerWithFit()
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
    component = MockEstimator()
    with pytest.raises(MethodPropertyNotFoundError):
        component.predict_proba(X)


def test_component_describe(test_classes, caplog):
    MockComponent, _, _ = test_classes
    component = MockComponent()
    component.describe(print_name=True)
    out = caplog.text
    assert "Mock Component" in out


def test_component_parameters_getter(test_classes):
    MockComponent, _, _ = test_classes
    component = MockComponent({'test': 'parameter'})
    assert component.parameters == {'test': 'parameter'}
    component.parameters['test'] = 'new'
    assert component.parameters == {'test': 'parameter'}


def test_component_parameters_init():
    components = all_components()
    for component_name, component_class in components.items():
        print('Testing component {}'.format(component_class.name))
        component = component_class()
        parameters = component.parameters

        component2 = component_class(**parameters)
        parameters2 = component2.parameters

        assert parameters == parameters2


def test_component_parameters_all_saved():
    components = all_components()
    for component_name, component_class in components.items():
        print('Testing component {}'.format(component_class.name))
        component = component_class()
        parameters = component.parameters

        spec = inspect.getfullargspec(component_class.__init__)
        args = spec.args
        assert args.pop(0) == 'self'
        defaults = list(spec.defaults)
        assert len(args) == len(defaults)
        # the last arg should always be random_state
        assert args.pop(-1) == 'random_state'
        assert defaults.pop(-1) == 0

        expected_parameters = {arg: default for (arg, default) in zip(args, defaults)}
        assert parameters == expected_parameters
