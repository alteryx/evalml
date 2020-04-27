from evalml.data_checks.data_check import DataCheck


def test_data_check():
    class MockDataCheck(DataCheck):
        def validate(X, y, verbose=True):
            return [], []
