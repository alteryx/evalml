class DataCheckAction:
    """Base class for all DataCheckActions."""

    def __init__(self, action_code, metadata=None):
        """
        A recommended action returned by a DataCheck.

        Arguments:
            action_code (DataCheckActionCode): Action code associated with the action.
            metadata (dict, optional): Additional useful information associated with the action. Defaults to None.
        """
        self.action_code = action_code
        self.metadata = metadata or {}

    def __eq__(self, other):
        """Checks for equality. Two DataCheckAction objs are considered equivalent if all of their attributes are equivalent."""
        return (self.action_code == other.action_code and
                self.metadata == other.metadata)

    def to_dict(self):
        action_dict = {
            "code": self.action_code.name,
            "metadata": self.metadata
        }
        return action_dict
