from evalml.pipelines.components import (
    ComponentBase,
    RFClassifierSelectFromModel,
    RFRegressorSelectFromModel
)


def make_rf_feature_selectors():
    rf_classifier = RFClassifierSelectFromModel(
        number_features=5,
        n_estimators=10,
        max_depth=7,
        percent_features=0.5,
        threshold=0,
    )
    rf_regressor = RFRegressorSelectFromModel(
        number_features=5,
        n_estimators=10,
        max_depth=7,
        percent_features=0.5,
        threshold=0,
    )
    return rf_classifier, rf_regressor


def test_init():
    rf_classifier, rf_regressor = make_rf_feature_selectors()
    assert rf_classifier.name == "RF Classifier Select From Model"
    assert rf_regressor.name == "RF Regressor Select From Model"


def test_component_fit(X_y_binary, X_y_regression):
    X, y = X_y_binary
    X_reg, y_reg = X_y_regression

    rf_classifier, rf_regressor = make_rf_feature_selectors()
    assert isinstance(rf_classifier.fit(X, y), ComponentBase)
    assert isinstance(rf_regressor.fit(X_reg, y_reg), ComponentBase)
