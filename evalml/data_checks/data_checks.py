"""A collection of data checks."""
import inspect

from evalml.data_checks import DataCheck
from evalml.exceptions import DataCheckInitError
from evalml.utils import infer_feature_types


def _has_defaults_for_all_args(init):
    """Test whether the init method has defaults for all arguments."""
    signature = inspect.getfullargspec(init)
    n_default_args = 0 if not signature.defaults else len(signature.defaults)
    n_args = (
        len(signature.args) - 1 if "self" in signature.args else len(signature.args)
    )
    return n_args == n_default_args


class DataChecks:
    """A collection of data checks.

    Args:
        data_checks (list (DataCheck)): List of DataCheck objects.
        data_check_params (dict): Parameters for passed DataCheck objects.
    """

    @staticmethod
    def _validate_data_checks(data_check_classes, params):
        """Creates a DataChecks instance from a list of DataCheck classes and corresponding params."""
        if not isinstance(data_check_classes, list):
            raise ValueError(
                f"Parameter data_checks must be a list. Received {type(data_check_classes).__name__}."
            )
        if not all(
            inspect.isclass(check) and issubclass(check, DataCheck)
            for check in data_check_classes
        ):
            raise ValueError(
                "All elements of parameter data_checks must be an instance of DataCheck "
                "or a DataCheck class with any desired parameters specified in the "
                "data_check_params dictionary."
            )
        params = params or dict()
        if not isinstance(params, dict):
            raise ValueError(f"Params must be a dictionary. Received {params}")
        in_params = set(params.keys())
        in_classes = set([c.name for c in data_check_classes])
        name_to_class = {c.name: c for c in data_check_classes}
        extraneous = in_params.difference(in_classes)
        missing = in_classes.difference(in_params)
        for extraneous_class in extraneous:
            raise DataCheckInitError(
                f"Class {extraneous_class} was provided in params dictionary but it does not match any name "
                "in the data_check_classes list. Make sure every key of the params dictionary matches the name"
                "attribute of a corresponding DataCheck class."
            )
        for missing_class_name in missing:
            if not _has_defaults_for_all_args(name_to_class[missing_class_name]):
                raise DataCheckInitError(
                    f"Class {missing_class_name} was provided in the data_checks_classes list but it does not have "
                    "an entry in the parameters dictionary."
                )

    @staticmethod
    def _init_data_checks(data_check_classes, params):
        data_check_instances = []
        for data_check_class in data_check_classes:
            class_params = params.get(data_check_class.name, {})
            if not isinstance(class_params, dict):
                raise DataCheckInitError(
                    f"Parameters for {data_check_class.name} were not in a dictionary. Received {class_params}."
                )
            try:
                data_check_instances.append(data_check_class(**class_params))
            except TypeError as e:
                raise DataCheckInitError(
                    f"Encountered the following error while initializing {data_check_class.name}: {e}"
                )
        return data_check_instances

    def __init__(self, data_checks=None, data_check_params=None):
        data_check_params = data_check_params or dict()
        self._validate_data_checks(data_checks, data_check_params)
        data_check_instances = self._init_data_checks(data_checks, data_check_params)
        self.data_checks = data_check_instances

    def validate(self, X, y=None):
        """Inspect and validate the input data against data checks and returns a list of warnings and errors if applicable.

        Args:
            X (pd.DataFrame, np.ndarray): The input data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): The target data of length [n_samples]

        Returns:
            dict: Dictionary containing DataCheckMessage objects
        """
        messages = {"warnings": [], "errors": [], "actions": []}
        X = infer_feature_types(X)
        X = X.ww.drop(list(X.ww.select("index", return_schema=True).columns))
        if y is not None:
            y = infer_feature_types(y)

        for data_check in self.data_checks:
            messages_new = data_check.validate(X, y)
            messages["warnings"].extend(messages_new["warnings"])
            messages["errors"].extend(messages_new["errors"])

            new_actions = messages_new["actions"]
            for new_action in new_actions:
                if new_action not in messages["actions"]:
                    messages["actions"].append(new_action)
        return messages
