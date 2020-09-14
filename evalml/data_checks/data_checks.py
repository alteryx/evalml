import inspect

from .data_check import DataCheck

from evalml.exceptions import DataCheckInitError


class DataChecks:
    """A collection of data checks."""

    def __init__(self, data_checks=None, data_check_params=None):
        """
        A collection of data checks.

        Arguments:
            data_checks (list (DataCheck)): List of DataCheck objects
        """
        if not isinstance(data_checks, list):
            raise ValueError(f"Parameter data_checks must be a list. Received {type(data_checks).__name__}.")
        if all(inspect.isclass(check) and issubclass(check, DataCheck) for check in data_checks):
            data_check_instances = init_data_checks_from_params(data_checks, data_check_params)
        elif all(isinstance(check, DataCheck) for check in data_checks):
            data_check_instances = data_checks
        else:
            raise ValueError("All elements of parameter data_checks must be an instance of DataCheck "
                             "or a DataCheck class with any desired parameters specified in the "
                             "data_check_params dictionary.")

        self.data_checks = data_check_instances

    def validate(self, X, y=None):
        """
        Inspects and validates the input data against data checks and returns a list of warnings and errors if applicable.

        Arguments:
            X (pd.DataFrame): The input data of shape [n_samples, n_features]
            y (pd.Series): The target data of length [n_samples]

        Returns:
            list (DataCheckMessage): List containing DataCheckMessage objects

        """
        messages = []
        for data_check in self.data_checks:
            messages_new = data_check.validate(X, y)
            messages.extend(messages_new)
        return messages


def init_data_checks_from_params(data_check_classes, params):
    """Inits a DataChecks instance from a list of DataCheck classes and corresponding params."""
    params = params or dict()
    if not isinstance(params, dict):
        raise ValueError(f"Params must be a dictionary. Received {params}")
    data_check_instances = []
    for extraneous_class in set(params.keys()).difference([c.name for c in data_check_classes]):
        raise DataCheckInitError(f"Class {extraneous_class} was provided in params dictionary but it does not match any name in "
                                 "in the data_check_classes list. Make sure every key of the params dictionary matches the name"
                                 "attribute of a corresponding DataCheck class.")

    for data_check_class in data_check_classes:
        class_params = params.get(data_check_class.name, {})
        if not isinstance(class_params, dict):
            raise DataCheckInitError(f"Parameters for {data_check_class.name} were not in a dictionary. Received {class_params}.")
        try:
            data_check_instances.append(data_check_class(**class_params))
        except TypeError as e:
            raise DataCheckInitError(f"Encountered the following error while initializing {data_check_class.name}: {e}")
    return data_check_instances
