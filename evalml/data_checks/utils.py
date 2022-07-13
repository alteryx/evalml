"""Utility methods for the data checks in EvalML."""
from evalml.data_checks.data_check_action_code import DataCheckActionCode


def handle_data_check_action_code(action_code):
    """Handles data check action codes by either returning the DataCheckActionCode or converting from a str.

    Args:
        action_code (str or DataCheckActionCode): Data check action code that needs to be handled.

    Returns:
        DataCheckActionCode enum

    Raises:
        KeyError: If input is not a valid DataCheckActionCode enum value.
        ValueError: If input is not a string or DatCheckActionCode object.

    Examples:
        >>> assert handle_data_check_action_code("drop_col") == DataCheckActionCode.DROP_COL
        >>> assert handle_data_check_action_code("DROP_ROWS") == DataCheckActionCode.DROP_ROWS
        >>> assert handle_data_check_action_code("Impute_col") == DataCheckActionCode.IMPUTE_COL
    """
    if isinstance(action_code, str):
        try:
            dcac = DataCheckActionCode._all_values[action_code.upper()]
        except KeyError:
            raise KeyError("Action code '{}' does not exist".format(action_code))
        return dcac
    if isinstance(action_code, DataCheckActionCode):
        return action_code
    raise ValueError(
        "`handle_data_check_action_code` was not passed a str or DataCheckActionCode object",
    )
