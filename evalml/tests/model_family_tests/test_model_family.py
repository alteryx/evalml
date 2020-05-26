import pytest

from evalml.model_family import ModelFamily, handle_model_family


@pytest.fixture
def correct_model_families():
    correct_model_families = [ModelFamily.LINEAR_MODEL, ModelFamily.RANDOM_FOREST, ModelFamily.XGBOOST, ModelFamily.CATBOOST, ModelFamily.ELASTIC_NET, ModelFamily.NONE]
    yield correct_model_families


def test_handle_string(correct_model_families):
    model_families = ['linear_model', 'random_forest', 'xgboost', 'catboost', 'elastic_net', 'none']
    for model_family in zip(model_families, correct_model_families):
        assert handle_model_family(model_family[0]) == model_family[1]

    model_family = 'fake'
    error_msg = 'Model family \'fake\' does not exist'
    with pytest.raises(KeyError, match=error_msg):
        handle_model_family(model_family) == ModelFamily.LINEAR_MODEL


def test_handle_model_family(correct_model_families):
    for model_family in correct_model_families:
        assert handle_model_family(model_family) == model_family


def test_handle_incorrect_type():
    error_msg = '`handle_model_family` was not passed a str or ModelFamily object'
    with pytest.raises(ValueError, match=error_msg):
        handle_model_family(5)
