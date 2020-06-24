import importlib
import inspect
from unittest.mock import patch

import numpy as np
import pandas as pd
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
from evalml.problem_types import ProblemTypes


@pytest.fixture(scope="module")
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


class MockFitComponent(ComponentBase):
    model_family = ModelFamily.NONE
    name = 'Mock Fit Component'

    def __init__(self, param_a=2, param_b=10, random_state=0):
        self.is_fitted = False
        parameters = {'param_a': param_a, 'param_b': param_b}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=0)

    def fit(self, X, y=None):
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Component is not fit')
        return np.array([self.parameters['param_a'] * 2, self.parameters['param_b'] * 10])


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
    assert enc.describe(return_dict=True) == {'name': 'One Hot Encoder', 'parameters': {'top_n': 10,
                                                                                        'categories': None,
                                                                                        'drop': None,
                                                                                        'handle_unknown': 'ignore',
                                                                                        'handle_missing': 'error'}}
    drop_col_transformer = DropColumns(columns=['col_one', 'col_two'])
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

    class MockEstimatorWithoutAttribute(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL

    with pytest.raises(TypeError):
        MockEstimatorWithoutAttribute()


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
            return self

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
            return self

        def __init__(self):
            parameters = {}
            super().__init__(parameters=parameters,
                             component_obj=None,
                             random_state=0)

    # convert data to pd DataFrame, because the component classes don't
    # standardize to pd DataFrame
    X = pd.DataFrame(X)
    y = pd.Series(y)

    component = MockTransformerWithFitTransform()
    assert isinstance(component.fit_transform(X, y), pd.DataFrame)

    component = MockTransformerWithFitTransformButError()
    with pytest.raises(RuntimeError):
        component.fit_transform(X, y)

    component = MockTransformerWithFitAndTransform()
    assert isinstance(component.fit_transform(X, y), pd.DataFrame)

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


def test_clone_init():
    params = {'param_a': 2, 'param_b': 11}
    clf = MockFitComponent(**params)
    clf_clone = clf.clone()
    assert clf.parameters == clf_clone.parameters


def test_clone_random_state():
    params = {'param_a': 1, 'param_b': 1}

    clf = MockFitComponent(**params, random_state=np.random.RandomState(42))
    clf_clone = clf.clone(random_state=np.random.RandomState(42))
    assert clf_clone.random_state.randint(2**30) == clf.random_state.randint(2**30)

    clf = MockFitComponent(**params, random_state=2)
    clf_clone = clf.clone(random_state=2)
    assert clf_clone.random_state.randint(2**30) == clf.random_state.randint(2**30)


def test_clone_fitted(X_y):
    X, y = X_y
    params = {'param_a': 3, 'param_b': 7}
    clf = MockFitComponent(**params)
    random_state_first_val = clf.random_state.randint(2**30)

    clf.fit(X, y)
    predicted = clf.predict(X)

    clf_clone = clf.clone()
    assert clf_clone.random_state.randint(2**30) == random_state_first_val
    with pytest.raises(ValueError, match='Component is not fit'):
        clf_clone.predict(X)
    assert clf.parameters == clf_clone.parameters

    clf_clone.fit(X, y)
    predicted_clone = clf_clone.predict(X)
    np.testing.assert_almost_equal(predicted, predicted_clone)


def test_components_init_kwargs():
    components = all_components()
    for component_name, component_class in components.items():
        component = component_class()
        if component._component_obj is None:
            continue

        obj_class = component._component_obj.__class__.__name__
        module = component._component_obj.__module__
        importlib.import_module(module, obj_class)
        patched = module + '.' + obj_class + '.__init__'

        def all_init(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        with patch(patched, new=all_init) as _:
            component = component_class(test_arg="test")
            assert component.parameters['test_arg'] == "test"
            assert component._component_obj.test_arg == "test"


def test_component_has_random_state():
    components = all_components()
    for component_name, component_class in components.items():
        params = inspect.signature(component_class.__init__).parameters
        assert "random_state" in params


def test_transformer_transform_output_type(X_y):
    X_np, y_np = X_y
    assert isinstance(X_np, np.ndarray)
    assert isinstance(y_np, np.ndarray)
    y_list = list(y_np)
    X_df_no_col_names = pd.DataFrame(X_np)
    range_index = pd.RangeIndex(start=0, stop=X_np.shape[1], step=1)
    X_df_with_col_names = pd.DataFrame(X_np, columns=['x' + str(i) for i in range(X_np.shape[1])])
    y_series_no_name = pd.Series(y_np)
    y_series_with_name = pd.Series(y_np, name='target')
    datatype_combos = [(X_np, y_np, range_index),
                       (X_np, y_list, range_index),
                       (X_df_no_col_names, y_series_no_name, range_index),
                       (X_df_with_col_names, y_series_with_name, X_df_with_col_names.columns)]

    components = all_components()
    transformers = dict(filter(lambda el: issubclass(el[1], Transformer), components.items()))
    for component_name, component_class in transformers.items():
        print('Testing transformer {}'.format(component_class.name))
        for X, y, X_cols_expected in datatype_combos:
            print('Checking output of transform for transformer "{}" on X type {} cols {}, y type {} name {}'
                  .format(component_class.name, type(X),
                          X.columns if isinstance(X, pd.DataFrame) else None, type(y),
                          y.name if isinstance(y, pd.Series) else None))
            component = component_class()
            component.fit(X, y=y)
            transform_output = component.transform(X, y=y)
            assert isinstance(transform_output, pd.DataFrame)
            assert transform_output.shape == X.shape
            assert (transform_output.columns == X_cols_expected).all()
            transform_output = component.fit_transform(X, y=y)
            assert isinstance(transform_output, pd.DataFrame)
            assert transform_output.shape == X.shape
            assert (transform_output.columns == X_cols_expected).all()


def test_estimator_predict_output_type(X_y):
    X_np, y_np = X_y
    assert isinstance(X_np, np.ndarray)
    assert isinstance(y_np, np.ndarray)
    y_list = list(y_np)
    X_df_no_col_names = pd.DataFrame(X_np)
    range_index = pd.RangeIndex(start=0, stop=X_np.shape[1], step=1)
    X_df_with_col_names = pd.DataFrame(X_np, columns=['x' + str(i) for i in range(X_np.shape[1])])
    y_series_no_name = pd.Series(y_np)
    y_series_with_name = pd.Series(y_np, name='target')
    datatype_combos = [(X_np, y_np, range_index, np.unique(y_np)),
                       (X_np, y_list, range_index, np.unique(y_np)),
                       (X_df_no_col_names, y_series_no_name, range_index, y_series_no_name.unique()),
                       (X_df_with_col_names, y_series_with_name, X_df_with_col_names.columns, y_series_with_name.unique())]

    components = all_components()
    estimators = dict(filter(lambda el: issubclass(el[1], Estimator), components.items()))
    for component_name, component_class in estimators.items():
        for X, y, X_cols_expected, y_cols_expected in datatype_combos:
            print('Checking output of predict for estimator "{}" on X type {} cols {}, y type {} name {}'
                  .format(component_class.name, type(X),
                          X.columns if isinstance(X, pd.DataFrame) else None, type(y),
                          y.name if isinstance(y, pd.Series) else None))
            component = component_class()
            component.fit(X, y=y)
            predict_output = component.predict(X)
            assert isinstance(predict_output, pd.Series)
            assert len(predict_output) == len(y)
            assert predict_output.name is None

            if not ((ProblemTypes.BINARY in component_class.supported_problem_types) or
                    (ProblemTypes.MULTICLASS in component_class.supported_problem_types)):
                continue
            print('Checking output of predict_proba for estimator "{}" on X type {} cols {}, y type {} name {}'
                  .format(component_class.name, type(X),
                          X.columns if isinstance(X, pd.DataFrame) else None, type(y),
                          y.name if isinstance(y, pd.Series) else None))
            predict_proba_output = component.predict_proba(X)
            assert isinstance(predict_proba_output, pd.DataFrame)
            assert predict_proba_output.shape == (len(y), len(np.unique(y)))
            assert (predict_proba_output.columns == y_cols_expected).all()
