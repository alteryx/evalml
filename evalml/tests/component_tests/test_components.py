from evalml.pipelines import (
    ComponentTypes,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    SelectFromModel,
    SimpleImputer,
    StandardScaler,
    XGBoostClassifier
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
    assert enc.component_type == ComponentTypes.ENCODER
    assert imputer.component_type == ComponentTypes.IMPUTER
    assert scaler.component_type == ComponentTypes.SCALER
    assert feature_selection.component_type == ComponentTypes.FEATURE_SELECTION

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
