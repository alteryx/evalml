class DataChecks:
    """A collection of data checks."""

    def __init__(self, data_checks=None):
        """
        A collection of data checks.

        Arguments:
            data_checks (list (DataCheck)): list of DataCheck objects
        """
        self.data_checks = data_checks

    def validate(self, X, y=None):
        """
        Inspects and validates the input data against data checks and return a list of warnings and errors if applicable.

        Arguments:
            X (pd.DataFrame): the input data of shape [n_samples, n_features]
            y (pd.Series): the target labels of length [n_samples]

        Returns:
            list (DataCheckMessage): list containing DataCheckMessage objects

        """
        errors_warnings = []
        for data_check in self.data_checks:
            errors_warnings_new = data_check.validate(X, y)
            errors_warnings.extend(errors_warnings_new)
        return errors_warnings
