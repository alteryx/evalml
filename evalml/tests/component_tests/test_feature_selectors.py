import pandas as pd
import pytest
import woodwork as ww
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Double,
    Integer,
    NaturalLanguage
)

from evalml.exceptions import MethodPropertyNotFoundError
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
            return self

    mock_feature_selector = MockFeatureSelector()
    mock_feature_selector.fit(pd.DataFrame(), pd.Series())
    with pytest.raises(MethodPropertyNotFoundError, match="Feature selector requires a transform method or a component_obj that implements transform"):
        mock_feature_selector.transform(pd.DataFrame())
    with pytest.raises(MethodPropertyNotFoundError, match="Feature selector requires a transform method or a component_obj that implements transform"):
        mock_feature_selector.fit_transform(pd.DataFrame())


def test_feature_selector_component_obj_missing_transform():
    class MockFeatureSelector(FeatureSelector):
        name = "Mock Feature Selector"

        def __init__(self):
            self._component_obj = None

        def fit(self, X, y):
            return self

    mock_feature_selector = MockFeatureSelector()
    mock_feature_selector.fit(pd.DataFrame(), pd.Series())
    with pytest.raises(MethodPropertyNotFoundError, match="Feature selector requires a transform method or a component_obj that implements transform"):
        mock_feature_selector.transform(pd.DataFrame())
    with pytest.raises(MethodPropertyNotFoundError, match="Feature selector requires a transform method or a component_obj that implements transform"):
        mock_feature_selector.fit_transform(pd.DataFrame())


@pytest.mark.parametrize("logical_type, X_df", [(ww.logical_types.Datetime, pd.DataFrame(pd.to_datetime(['20190902', '20200519', '20190607'], format='%Y%m%d'))),
                                                (ww.logical_types.Integer, pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64"))),
                                                (ww.logical_types.Double, pd.DataFrame(pd.Series([1., 2., 3.], dtype="float"))),
                                                (ww.logical_types.Categorical, pd.DataFrame(pd.Series(['a', 'b', 'a'], dtype="category"))),
                                                (ww.logical_types.NaturalLanguage, pd.DataFrame(pd.Series(['this will be a natural language column because length', 'yay', 'hay'], dtype="string"))),
])
def test_fs_woodwork_custom_overrides_returned_by_components(logical_type, X_df):
    rf_classifier, rf_regressor = make_rf_feature_selectors()
    y = pd.Series([1, 2, 1])
    X_df['l'] = pd.Series([1., 2., 3.], dtype="float")
    types_to_test = [Integer, Double, ]
    for l in types_to_test:
        X = None
        override_dict = {0: l}
        try:
            X = ww.DataTable(X_df, logical_types=override_dict)
            assert X.logical_types[0] == l
        except TypeError:
            continue
        print ("testing override", logical_type, "with", l)

        rf_classifier.fit(X, y)
        transformed = rf_classifier.transform(X, y)
        assert isinstance(transformed, ww.DataTable)
        input_logical_types = {0: l}
        print ("transformed", transformed.logical_types.items())
        print ("expected", input_logical_types.items())

        assert transformed.logical_types == {0: l, 'l': Double}
