class DataChecks:
    def __init__(self, data_checks):
        self.data_checks = data_checks

    def validate(self, X, y, verbose=True):
        # call validate for each of the data checks and accumulate messages
        errors, warnings = [], []
        for data_check in self.data_checks:
            errors_new, warnings_new = data_check.validate(X, y, verbose=verbose)
            errors += errors_new
            warnings += warnings_new
        return errors, warnings
