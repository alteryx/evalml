class DataChecks:
    """A collection of data checks."""

    def __init__(self, data_checks):
        """
        A collection of data checks.

        Arguments:
            data_checks (list (DataCheck)): list of DataCheck objects
        """
        self.data_checks = data_checks

    def validate(self, X, y=None, verbose=True):
        """
        Inspects and validates the input data against data checks and return a list of warnings and errors if applicable.

        Arguments:
            X (pd.DataFrame): the input data of shape [n_samples, n_features]
            y (pd.Series): the target labels of length [n_samples]
            verbose (bool): Controls verbosity of output. If True, prints to console.

        Returns:
            dict: returns a dictionary containing DataCheckError and DataCheckWarning objects

        """
        errors_and_warnings = []
        for data_check in self.data_checks:
            errors_new, warnings_new = data_check.validate(X, y, verbose=verbose)
            errors_and_warnings.extend(errors_new)
            errors_and_warnings.extend(warnings_new)
        return errors_and_warnings
