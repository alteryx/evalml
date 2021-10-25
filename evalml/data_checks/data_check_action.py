"""Recommended action returned by a DataCheck."""
from types import MethodWrapperType

from evalml.data_checks.data_check_action_code import DataCheckActionCode


class DataCheckAction:
    """A recommended action returned by a DataCheck.

    Args:
        action_code (DataCheckActionCode): Action code associated with the action.
        metadata (dict, optional): Additional useful information associated with the action. Defaults to None.
    """

    def __init__(self, action_code, metadata=None):
        self.action_code = action_code
        self.metadata = {"columns": None, "rows": None}
        if metadata is not None:
            self.metadata.update(metadata)

    def __eq__(self, other):
        """Check for equality.

        Two DataCheckAction objs are considered equivalent if all of their attributes are equivalent.

        Args:
            other: An object to compare equality with.

        Returns:
            bool: True if the other object is considered an equivalent data check action, False otherwise.
        """
        return self.action_code == other.action_code and self.metadata == other.metadata

    def to_dict(self):
        """Return a dictionary form of the data check action."""
        action_dict = {"code": self.action_code.name, "metadata": self.metadata}
        return action_dict

    @staticmethod
    def convert_dict_to_action(action_dict):
        return DataCheckAction(
            action_code=DataCheckActionCode._all_values[action_dict["code"]],
            metadata=action_dict["metadata"],
        )
