"""Recommended action returned by a DataCheck."""

from evalml.data_checks.data_check_action_code import DataCheckActionCode
from evalml.data_checks.utils import handle_data_check_action_code


class DataCheckAction:
    """A recommended action returned by a DataCheck.

    Args:
        action_code (str, DataCheckActionCode): Action code associated with the action.
        data_check_name (str): Name of data check.
        metadata (dict, optional): Additional useful information associated with the action. Defaults to None.
    """

    def __init__(self, action_code, data_check_name, metadata=None):
        self.action_code = handle_data_check_action_code(action_code)
        self.data_check_name = data_check_name
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
        attributes_to_check = ["action_code", "data_check_name", "metadata"]
        for attribute in attributes_to_check:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def to_dict(self):
        """Return a dictionary form of the data check action."""
        action_dict = {
            "code": self.action_code.name,
            "data_check_name": self.data_check_name,
            "metadata": self.metadata,
        }
        return action_dict

    @staticmethod
    def convert_dict_to_action(action_dict):
        """Convert a dictionary into a DataCheckAction.

        Args:
            action_dict: Dictionary to convert into action. Should have keys "code", "data_check_name", and "metadata".

        Raises:
            ValueError: If input dictionary does not have keys `code` and `metadata` and if the `metadata` dictionary does not have keys `columns` and `rows`.

        Returns:
            DataCheckAction object from the input dictionary.
        """
        if "code" not in action_dict or "metadata" not in action_dict:
            raise ValueError(
                "The input dictionary should have the keys `code` and `metadata`.",
            )
        if (
            "columns" not in action_dict["metadata"]
            or "rows" not in action_dict["metadata"]
        ):
            raise ValueError(
                "The metadata dictionary should have the keys `columns` or `rows`. Set to None if not using.",
            )

        return DataCheckAction(
            action_code=DataCheckActionCode._all_values[action_dict["code"]],
            metadata=action_dict["metadata"],
            data_check_name=action_dict["data_check_name"]
            if "data_check_name" in action_dict
            else None,
        )
