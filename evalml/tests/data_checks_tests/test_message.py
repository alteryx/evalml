from evalml.data_checks.messsage import (
    DataCheckError,
    DataCheckWarning,
    Message
)


def test_data_check_message():
    m = Message("test message")
    assert m.message == "test message"
    assert str(m) == "test message"

    data_warning = DataCheckWarning("test warning")
    assert data_warning.message == "test warning"
    assert str(data_warning) == "test warning"

    data_error = DataCheckError("test error")
    assert data_error.message == "test error"
    assert str(data_error) == "test error"
