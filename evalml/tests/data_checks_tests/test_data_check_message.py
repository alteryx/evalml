import pytest

from evalml.data_checks import (
    DataCheckError,
    DataCheckMessage,
    DataCheckMessageCode,
    DataCheckMessageType,
    DataCheckWarning
)


@pytest.fixture
def data_check_message():
    return DataCheckMessage(message="test message",
                            data_check_name="test data check message name",
                            message_code=DataCheckMessageCode.HIGHLY_NULL)


@pytest.fixture
def data_check_warning():
    return DataCheckWarning(message="test warning",
                            data_check_name="test data check warning name",
                            message_code=DataCheckMessageCode.HIGHLY_NULL)


@pytest.fixture
def data_check_error():
    return DataCheckError(message="test error",
                          data_check_name="test data check error name",
                          message_code=DataCheckMessageCode.HIGHLY_NULL)


def test_data_check_message_attributes(data_check_message):
    assert data_check_message.message == "test message"
    assert data_check_message.data_check_name == "test data check message name"
    assert data_check_message.message_type is None
    assert data_check_message.message_code == DataCheckMessageCode.HIGHLY_NULL


def test_data_check_message_str(data_check_message):
    assert str(data_check_message) == "test message"


def test_data_check_message_eq(data_check_message):
    equal_msg = DataCheckMessage("test message", "test data check message name", DataCheckMessageCode.HIGHLY_NULL)
    assert data_check_message == equal_msg

    equal_msg = DataCheckMessage("different test message", "different test data check message name")
    assert data_check_message != equal_msg


def test_data_check_warning_attributes(data_check_warning):
    assert data_check_warning.message == "test warning"
    assert data_check_warning.data_check_name == "test data check warning name"
    assert data_check_warning.message_type == DataCheckMessageType.WARNING
    assert data_check_warning.message_code == DataCheckMessageCode.HIGHLY_NULL


def test_data_check_warning_str(data_check_warning):
    assert str(data_check_warning) == "test warning"


def test_data_check_warning_eq(data_check_warning):
    equal_msg = DataCheckWarning("test warning", "test data check warning name", DataCheckMessageCode.HIGHLY_NULL)
    assert data_check_warning == equal_msg

    equal_msg = DataCheckWarning("different test warning", "different test data check warning name")
    assert data_check_warning != equal_msg


def test_data_check_error_attributes(data_check_error):
    assert data_check_error.message == "test error"
    assert data_check_error.data_check_name == "test data check error name"
    assert data_check_error.message_type == DataCheckMessageType.ERROR
    assert data_check_error.message_code == DataCheckMessageCode.HIGHLY_NULL


def test_data_check_error_str(data_check_error):
    assert str(data_check_error) == "test error"


def test_data_check_error_eq(data_check_error):
    equal_msg = DataCheckError("test error", "test data check error name", DataCheckMessageCode.HIGHLY_NULL)
    assert data_check_error == equal_msg

    equal_msg = DataCheckError("different test warning", "different test data check error name")
    assert data_check_error != equal_msg


def test_warning_error_eq():
    error = DataCheckError("test message", "same test name")
    warning = DataCheckWarning("test message", "same test name")
    assert error != warning


def test_data_check_message_to_dict():
    error = DataCheckError("test message", "same test name", DataCheckMessageCode.HIGHLY_NULL)
    assert error.to_dict() == {"message": "test message", "level": "error", "data_check_name": "same test name", "code": DataCheckMessageCode.HIGHLY_NULL}
    warning = DataCheckWarning("test message", "same test name", DataCheckMessageCode.HIGHLY_NULL)
    assert warning.to_dict() == {"message": "test message", "level": "warning", "data_check_name": "same test name", "code": DataCheckMessageCode.HIGHLY_NULL}
