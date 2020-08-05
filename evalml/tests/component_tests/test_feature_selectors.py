import pandas as pd
import pytest

from evalml.pipelines.components import (
    ComponentBase,
    FeatureSelector,
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


def test_component_fit(X_y_binary, X_y_multi, X_y_regression):
    X_binary, y_binary = X_y_binary
    X_multi, y_multi = X_y_multi
    X_reg, y_reg = X_y_regression

    rf_classifier, rf_regressor = make_rf_feature_selectors()
    assert isinstance(rf_classifier.fit(X_binary, y_binary), ComponentBase)
    assert isinstance(rf_classifier.fit(X_multi, y_multi), ComponentBase)
    assert isinstance(rf_regressor.fit(X_reg, y_reg), ComponentBase)


def test_feature_selector_missing_component_obj():
    class MockFeatureSelector(FeatureSelector):
        name = "Mock Feature Selector"

        def fit(self, X, y):
            pass

    mock_feature_selector = MockFeatureSelector()
    mock_feature_selector.fit(pd.DataFrame(), pd.Series())
    with pytest.raises(RuntimeError, match="Transformer requires a transform method or a component_obj that implements transform"):
        mock_feature_selector.transform(pd.DataFrame())
    with pytest.raises(RuntimeError, match="Transformer requires a fit_transform method or a component_obj that implements fit_transform"):
        mock_feature_selector.fit_transform(pd.DataFrame())
