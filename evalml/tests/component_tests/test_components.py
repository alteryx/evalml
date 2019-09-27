from sklearn.ensemble import RandomForestClassifier

from evalml.pipelines.components import (
    OneHotEncoder,
    SelectFromModel,
    SimpleImputer,
    StandardScaler
)


def test_init():
    enc = OneHotEncoder()
    imputer = SimpleImputer()
    scaler = StandardScaler()
    feature_selection = SelectFromModel(estimator=RandomForestClassifier(), number_features=5)

    assert enc.component_type == 'encoder'
    assert imputer.component_type == 'imputer'
    assert scaler.component_type == 'scaler'
    assert feature_selection.component_type == 'feature_selection'
