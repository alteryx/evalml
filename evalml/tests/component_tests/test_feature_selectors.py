import numpy as np
import pandas as pd
import pytest

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


def test_component_fit(X_y):
    X, y = X_y

    rf_classifier, rf_regressor = make_rf_feature_selectors()
    assert isinstance(rf_classifier.fit(X, y), ComponentBase)
    assert isinstance(rf_regressor.fit(X, y), ComponentBase)


def test_clone(X_y):
    X = pd.DataFrame()
    X["col_1"] = [2, 0, 1, 0, 0]
    X["col_2"] = [3, 2, 5, 1, 3]
    X["col_3"] = [0, 0, 1, 3, 2]
    X["col_4"] = [2, 4, 1, 4, 0]
    y = [0, 1, 0, 1, 1]
    rf_classifier, rf_regressor = make_rf_feature_selectors()

    rf_classifier.fit(X, y)
    rf_regressor.fit(X, y)
    transformed_classifier = rf_classifier.transform(X).values
    transformed_regressor = rf_regressor.transform(X).values
    assert isinstance(transformed_classifier, type(np.array([])))
    assert isinstance(transformed_regressor, type(np.array([])))

    clf_clone = rf_classifier.clone(learned=False)
    reg_clone = rf_regressor.clone(learned=False)
    with pytest.raises(ValueError):
        clf_clone.transform(X)
    with pytest.raises(ValueError):
        reg_clone.transform(X)

    clf_clone.fit(X, y)
    reg_clone.fit(X, y)
    transformed_clf_clone = clf_clone.transform(X)
    transformed_reg_clone = reg_clone.transform(X)
    np.testing.assert_almost_equal(transformed_classifier, transformed_clf_clone)
    np.testing.assert_almost_equal(transformed_regressor, transformed_reg_clone)
