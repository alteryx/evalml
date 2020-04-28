class DataChecks:
    def __init__(self, data_checks):
        """
        A collection of data checks.

        Arguments:
            data_checks (list): list of data checks
        """
        self.data_checks = data_checks

    def validate(self, X, y, verbose=True):
        """
        Inspects and validates the input data against data checks and return a list of warnings and errors if applicable.

        Arguments:
            X (pd.DataFrame): the input data of shape [n_samples, n_features]
            y (pd.Series): the target labels of length [n_samples]
            verbose (bool): Controls verbosity of output. If True, prints to console.

        Returns:
            list (DataCheckError), list (DataCheckWarning): returns a list of DataCheckError and DataCheckWarning objects

        """
        # call validate for each of the data checks and accumulate messages
        errors, warnings = [], []
        for data_check in self.data_checks:
            errors_new, warnings_new = data_check.validate(X, y, verbose=verbose)
            errors += errors_new
            warnings += warnings_new
        return errors, warnings
