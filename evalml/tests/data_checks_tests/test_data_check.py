from evalml.data_checks.data_check import DataCheck


def test_empty_data_check(X_y):
    X, y = X_y

    class MockDataCheck(DataCheck):
        def validate(X, y, verbose=True):
            return [], []

    data_check = MockDataCheck()
    errors, warnings = data_check.validate()
    assert len(errors) == 0
    assert len(warnings) == 0
