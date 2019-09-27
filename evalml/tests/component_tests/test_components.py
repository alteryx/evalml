from evalml.pipelines import (
    LinearRegressor,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    XGBoostClassifier
)
from evalml.pipelines.components import (
    OneHotEncoder,
    SelectFromModel,
    SimpleImputer,
    StandardScaler
)

# Tests to include:
#   for each specific estimator
#   for an user-defined estimator


def test_init():
    # testing transformers
    enc = OneHotEncoder()
    imputer = SimpleImputer()
    scaler = StandardScaler()
    feature_selection = SelectFromModel(estimator=RandomForestClassifier(n_estimators=10), number_features=5)
    assert enc.component_type == 'encoder'
    assert imputer.component_type == 'imputer'
    assert scaler.component_type == 'scaler'
    assert feature_selection.component_type == 'feature_selection'

    # testing estimators
    lr_classifier = LogisticRegressionClassifier()
    rf_classifier = RandomForestClassifier(n_estimators=10)
    xgb_classifier = XGBoostClassifier(eta=0.1, min_child_weight=1, max_depth=3)
    rf_regressor = RandomForestRegressor(n_estimators=10)
    linear_regressor = LinearRegressor()
    assert lr_classifier.component_type == 'classifier'
    assert rf_classifier.component_type == 'classifier'
    assert xgb_classifier.component_type == 'classifier'
    assert rf_regressor.component_type == 'regressor'
    assert linear_regressor.component_type == 'regressor'
