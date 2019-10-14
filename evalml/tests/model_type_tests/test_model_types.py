import pytest

from evalml.model_types import ModelTypes, handle_model_types


@pytest.fixture
def correct_model_types():
    correct_model_types = [ModelTypes.LINEAR_MODEL, ModelTypes.RANDOM_FOREST, ModelTypes.XGBOOST]
    yield correct_model_types


def test_handle_string(correct_model_types):
    model_types = ['linear_model', 'random_forest', 'xgboost']
    for model_type in zip(model_types, correct_model_types):
        assert handle_model_types(model_type[0]) == model_type[1]

    model_type = 'fake'
    error_msg = 'Model type \'fake\' does not exist'
    with pytest.raises(KeyError, match=error_msg):
        handle_model_types(model_type) == ModelTypes.LINEAR_MODEL


def test_handle_model_types(correct_model_types):
    for modle_type in correct_model_types:
        assert handle_model_types(modle_type) == modle_type


def test_handle_incorrect_type():
    error_msg = '`handle_model_types` was not passed a str or ModelTypes object'
    with pytest.raises(ValueError, match=error_msg):
        handle_model_types(5)
