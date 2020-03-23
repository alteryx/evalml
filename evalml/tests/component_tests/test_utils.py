import pytest

from evalml.pipelines.components import handle_component
from evalml.pipelines.components.utils import __COMPONENTS, __components_dict


def test_components_dict():
    components_dict = __components_dict()
    assert len(components_dict) == 12
    names = list(components_dict.keys())
    names.sort()
    assert names == [
        'CatBoost Classifier',
        'CatBoost Regressor',
        'Linear Regressor',
        'Logistic Regression Classifier',
        'One Hot Encoder',
        'RF Classifier Select From Model',
        'RF Regressor Select From Model',
        'Random Forest Classifier',
        'Random Forest Regressor',
        'Simple Imputer',
        'Standard Scaler',
        'XGBoost Classifier'
    ]


def test_global_components():
    assert __COMPONENTS == __components_dict()


def test_handle_component_names():
    for name, cls in __COMPONENTS.items():
        assert isinstance(handle_component(cls()), cls)
        assert isinstance(handle_component(name), cls)

    invalid_name = 'This Component Does Not Exist'
    with pytest.raises(ValueError):
        handle_component(invalid_name)

    class NonComponent:
        pass
    with pytest.raises(ValueError):
        handle_component(NonComponent())
