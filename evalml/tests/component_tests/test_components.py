import pytest

from evalml.pipelines import (
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
from evalml.pipelines.components import ComponentBase, ComponentTypes


def test_init():
    # testing transformers
    enc = OneHotEncoder()
    imputer = SimpleImputer()
    scaler = StandardScaler()
    feature_selection = RFClassifierSelectFromModel(n_estimators=10, number_features=5)
    assert enc.component_type == ComponentTypes.CATEGORICAL_ENCODER
    assert imputer.component_type == ComponentTypes.IMPUTER
    assert scaler.component_type == ComponentTypes.SCALER
    assert feature_selection.component_type == ComponentTypes.FEATURE_SELECTION_CLASSIFIER

    # testing estimators
    lr_classifier = LogisticRegressionClassifier()
    rf_classifier = RandomForestClassifier(n_estimators=10)
    xgb_classifier = XGBoostClassifier(eta=0.1, min_child_weight=1, max_depth=3)
    rf_regressor = RandomForestRegressor(n_estimators=10)
    linear_regressor = LinearRegressor()
    assert lr_classifier.component_type == ComponentTypes.CLASSIFIER
    assert rf_classifier.component_type == ComponentTypes.CLASSIFIER
    assert xgb_classifier.component_type == ComponentTypes.CLASSIFIER
    assert rf_regressor.component_type == ComponentTypes.REGRESSOR
    assert linear_regressor.component_type == ComponentTypes.REGRESSOR


def test_describe_component():
    enc = OneHotEncoder()
    imputer = SimpleImputer("mean")
    scaler = StandardScaler()
    feature_selection = RFClassifierSelectFromModel(n_estimators=10, number_features=5, percent_features=0.3, threshold=10)
    assert enc.describe(return_dict=True) == {'name': 'One Hot Encoder', 'parameters': {}}
    assert imputer.describe(return_dict=True) == {'name': 'Simple Imputer', 'parameters': {'impute_strategy': 'mean'}}
    assert scaler.describe(return_dict=True) == {'name': 'Standard Scaler', 'parameters': {}}
    assert feature_selection.describe(return_dict=True) == {'name': 'RF Classifier Select From Model', 'parameters': {'percent_features': 0.3, 'threshold': 10}}

    # testing estimators
    lr_classifier = LogisticRegressionClassifier()
    rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=3)
    xgb_classifier = XGBoostClassifier(eta=0.1, min_child_weight=1, max_depth=3, n_estimators=75)
    rf_regressor = RandomForestRegressor(n_estimators=10, max_depth=3)
    linear_regressor = LinearRegressor()
    assert lr_classifier.describe(return_dict=True) == {'name': 'Logistic Regression Classifier', 'parameters': {'C': 1.0, 'penalty': 'l2'}}
    assert rf_classifier.describe(return_dict=True) == {'name': 'Random Forest Classifier', 'parameters': {'max_depth': 3, 'n_estimators': 10}}
    assert xgb_classifier.describe(return_dict=True) == {'name': 'XGBoost Classifier', 'parameters': {'eta': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 75}}
    assert rf_regressor.describe(return_dict=True) == {'name': 'Random Forest Regressor', 'parameters': {'max_depth': 3, 'n_estimators': 10}}
    assert linear_regressor.describe(return_dict=True) == {'name': 'Linear Regressor', 'parameters': {'fit_intercept': True, 'normalize': False}}


def test_missing_attributes(X_y):
    class mockComponentFitting(ComponentBase):
        name = "mock"
        component_type = ComponentTypes.REGRESSOR

    class mockComponentName(ComponentBase):
        component_type = ComponentTypes.REGRESSOR
        _needs_fitting = True

    class mockComponentType(ComponentBase):
        name = "mock"
        _needs_fitting = True

    with pytest.raises(AttributeError, match="Component missing attribute: `name`"):
        mockComponentName(parameters={}, component_obj=None, random_state=0)

    with pytest.raises(AttributeError, match="Component missing attribute: `_needs_fitting`"):
        mockComponentFitting(parameters={}, component_obj=None, random_state=0)

    with pytest.raises(AttributeError, match="Component missing attribute: `component_type`"):
        mockComponentType(parameters={}, component_obj=None, random_state=0)


def test_missing_methods_on_components(X_y):
    # test that estimator doesn't have
    X, y = X_y

    class mockEstimator(Estimator):
        name = "mock Estimator"
        component_type = ComponentTypes.REGRESSOR
        _needs_fitting = True

    class mockTransformer(Transformer):
        name = "mock Transformer"
        component_type = ComponentTypes.IMPUTER
        _needs_fitting = False

    estimator = mockEstimator(parameters={}, component_obj=None, random_state=0)
    with pytest.raises(RuntimeError, match="Estimator requires a predict method or a component_obj that implements predict"):
        estimator.predict(X)
    with pytest.raises(RuntimeError, match="Estimator requires a predict_proba method or a component_obj that implements predict_proba"):
        estimator.predict_proba(X)

    transformer = mockTransformer(parameters={}, component_obj=None, random_state=0)
    with pytest.raises(RuntimeError, match="Component requires a fit method or a component_obj that implements fit"):
        transformer.fit(X, y)
    with pytest.raises(RuntimeError, match="Transformer requires a transform method or a component_obj that implements transform"):
        transformer.transform(X)
    with pytest.raises(RuntimeError, match="Transformer requires a fit_transform method or a component_obj that implements fit_transform"):
        transformer.fit_transform(X)
