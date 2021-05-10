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
                            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                            details={"message detail": "some message detail"})


@pytest.fixture
def data_check_warning():
    return DataCheckWarning(message="test warning",
                            data_check_name="test data check warning name",
                            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                            details={"warning detail": "some warning detail"})


@pytest.fixture
def data_check_error():
    return DataCheckError(message="test error",
                          data_check_name="test data check error name",
                          message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                          details={"error detail": "some error detail"})


def test_data_check_message_attributes(data_check_message):
    assert data_check_message.message == "test message"
    assert data_check_message.data_check_name == "test data check message name"
    assert data_check_message.message_type is None
    assert data_check_message.message_code == DataCheckMessageCode.HIGHLY_NULL_COLS
    assert data_check_message.details == {"message detail": "some message detail"}


def test_data_check_message_str(data_check_message):
    assert str(data_check_message) == "test message"


def test_data_check_message_eq(data_check_message):
    equal_msg = DataCheckMessage("test message", "test data check message name", DataCheckMessageCode.HIGHLY_NULL_COLS, {"message detail": "some message detail"})
    assert data_check_message == equal_msg

    equal_msg = DataCheckMessage("different test message", "different test data check message name")
    assert data_check_message != equal_msg


def test_data_check_warning_attributes(data_check_warning):
    assert data_check_warning.message == "test warning"
    assert data_check_warning.data_check_name == "test data check warning name"
    assert data_check_warning.message_type == DataCheckMessageType.WARNING
    assert data_check_warning.message_code == DataCheckMessageCode.HIGHLY_NULL_COLS
    assert data_check_warning.details == {"warning detail": "some warning detail"}


def test_data_check_warning_str(data_check_warning):
    assert str(data_check_warning) == "test warning"


def test_data_check_warning_eq(data_check_warning):
    equal_msg = DataCheckWarning("test warning", "test data check warning name", DataCheckMessageCode.HIGHLY_NULL_COLS, {"warning detail": "some warning detail"})
    assert data_check_warning == equal_msg

    equal_msg = DataCheckWarning("different test warning", "different test data check warning name")
    assert data_check_warning != equal_msg


def test_data_check_error_attributes(data_check_error):
    assert data_check_error.message == "test error"
    assert data_check_error.data_check_name == "test data check error name"
    assert data_check_error.message_type == DataCheckMessageType.ERROR
    assert data_check_error.message_code == DataCheckMessageCode.HIGHLY_NULL_COLS
    assert data_check_error.details == {"error detail": "some error detail"}


def test_data_check_error_str(data_check_error):
    assert str(data_check_error) == "test error"


def test_data_check_error_eq(data_check_error):
    equal_msg = DataCheckError("test error", "test data check error name", DataCheckMessageCode.HIGHLY_NULL_COLS, {"error detail": "some error detail"})
    assert data_check_error == equal_msg

    equal_msg = DataCheckError("different test warning", "different test data check error name")
    assert data_check_error != equal_msg


def test_data_check_message_attributes_optional():
    data_check_warning = DataCheckWarning(message="test warning",
                                          data_check_name="test data check warning name")
    assert data_check_warning.message == "test warning"
    assert data_check_warning.data_check_name == "test data check warning name"
    assert data_check_warning.message_type == DataCheckMessageType.WARNING
    assert data_check_warning.message_code is None
    assert data_check_warning.details is None

    data_check_error = DataCheckError(message="test error",
                                      data_check_name="test data check error name")
    assert data_check_error.message == "test error"
    assert data_check_error.data_check_name == "test data check error name"
    assert data_check_error.message_type == DataCheckMessageType.ERROR
    assert data_check_error.message_code is None
    assert data_check_error.details is None


def test_warning_error_eq():
    error = DataCheckError("test message", "same test name")
    warning = DataCheckWarning("test message", "same test name")
    assert error != warning


def test_data_check_message_to_dict():
    error = DataCheckError(message="test message",
                           data_check_name="same test name",
                           message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                           details={"detail 1": "error info"})
    assert error.to_dict() == {
        "message": "test message",
        "level": "error",
        "data_check_name": "same test name",
        "code": DataCheckMessageCode.HIGHLY_NULL_COLS.name,
        "details": {"detail 1": "error info"}
    }
    warning = DataCheckWarning(message="test message",
                               data_check_name="same test name",
                               message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                               details={"detail 1": "warning info"})
    assert warning.to_dict() == {
        "message": "test message",
        "level": "warning",
        "data_check_name": "same test name",
        "code": DataCheckMessageCode.HIGHLY_NULL_COLS.name,
        "details": {"detail 1": "warning info"}
    }


def test_data_check_message_to_dict_optional():
    error = DataCheckError(message="test message",
                           data_check_name="same test name")
    assert error.to_dict() == {
        "message": "test message",
        "level": "error",
        "data_check_name": "same test name"
    }
    warning = DataCheckWarning(message="test message",
                               data_check_name="same test name")
    assert warning.to_dict() == {
        "message": "test message",
        "level": "warning",
        "data_check_name": "same test name"
    }
