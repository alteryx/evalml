import pytest

from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckMessage,
    DataCheckWarning
)
from evalml.data_checks.data_check_message_type import DataCheckMessageType


@pytest.fixture
def data_check_message():
    message_str = "test message"
    data_check_name = "test data check message name"
    return DataCheckMessage(message_str, data_check_name)


@pytest.fixture
def data_check_warning():
    message_str = "test warning"
    data_check_name = "test data check warning name"
    return DataCheckWarning(message_str, data_check_name)


@pytest.fixture
def data_check_error():
    message_str = "test error"
    data_check_name = "test data check error name"
    return DataCheckError(message_str, data_check_name)


def test_data_check_message_attributes(data_check_message):
    assert data_check_message.message == "test message"
    assert data_check_message.data_check_name == "test data check message name"
    assert data_check_message.message_type is None


def test_data_check_message_str(data_check_message):
    assert str(data_check_message) == "test message"


def test_data_check_message_eq(data_check_message):
    message_str_dup = "test message"
    data_check_name_dup = "test data check message name"
    equal_msg = DataCheckMessage(message_str_dup, data_check_name_dup)
    assert data_check_message == equal_msg

    message_str_diff = "different test message"
    data_check_name_diff = "different test data check message name"
    equal_msg = DataCheckMessage(message_str_diff, data_check_name_diff)
    assert not data_check_message == equal_msg


def test_data_check_warning_attributes(data_check_warning):
    assert data_check_warning.message == "test warning"
    assert data_check_warning.data_check_name == "test data check warning name"
    assert data_check_warning.message_type == DataCheckMessageType.WARNING


def test_data_check_warning_str(data_check_warning):
    assert str(data_check_warning) == "test warning"


def test_data_check_warning_eq(data_check_warning):
    message_str_dup = "test warning"
    data_check_name_dup = "test data check warning name"
    equal_msg = DataCheckWarning(message_str_dup, data_check_name_dup)
    assert data_check_warning == equal_msg

    message_str_diff = "different test warning"
    data_check_name_diff = "different test data check warning name"
    equal_msg = DataCheckWarning(message_str_diff, data_check_name_diff)
    assert not data_check_warning == equal_msg


def test_data_check_error_attributes(data_check_error):
    assert data_check_error.message == "test error"
    assert data_check_error.data_check_name == "test data check error name"
    assert data_check_error.message_type == DataCheckMessageType.ERROR


def test_data_check_error_str(data_check_error):
    assert str(data_check_error) == "test error"


def test_data_check_error_eq(data_check_error):
    message_str_dup = "test error"
    data_check_name_dup = "test data check error name"
    equal_msg = DataCheckError(message_str_dup, data_check_name_dup)
    assert data_check_error == equal_msg

    message_str_diff = "different test warning"
    data_check_name_diff = "different test data check error name"
    equal_msg = DataCheckError(message_str_diff, data_check_name_diff)
    assert not data_check_error == equal_msg
