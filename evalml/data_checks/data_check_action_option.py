"""Recommended action returned by a DataCheck."""

from evalml.data_checks.data_check_action_code import DataCheckActionCode


class DataCheckActionOption:
    """A recommended action option returned by a DataCheck.

        It contains an action code that indicates what the
        action should be, a data check name that indicates what data check was used to generate the action, and
        parameters and metadata which can be used to further refine the action.

    Args:
        action_code (DataCheckActionCode): Action code associated with the action option.
        data_check_name (str): Name of the data check that produced this option.
        parameters (dict): Parameters associated with the action option. Defaults to None.
        metadata (dict, optional): Additional useful information associated with the action option. Defaults to None.


    Examples:
        >>> parameters = {
        ...     "global_parameter_name": {
        ...         "parameter_type": "global",
        ...         "type": "float",
        ...         "default_value": 0.0,
        ...     },
        ...     "column_parameter_name": {
        ...         "parameter_type": "column",
        ...         "columns": {
        ...             "a": {
        ...                 "impute_strategy": {
        ...                     "categories": ["mean", "mode"],
        ...                     "type": "category",
        ...                     "default_value": "mean",
        ...                 },
        ...             "constant_fill_value": {"type": "float", "default_value": 0},
        ...             },
        ...         },
        ...     },
        ... }
        >>> data_check_action = DataCheckActionOption(DataCheckActionCode.DROP_COL, None, metadata={}, parameters=parameters)


    """

    def __init__(self, action_code, data_check_name, parameters=None, metadata=None):
        self.action_code = action_code
        self.data_check_name = data_check_name
        self.parameters = parameters
        self.metadata = {"columns": None, "rows": None}
        if metadata is not None:
            self.metadata.update(metadata)
        self._validate_parameters()

    def __eq__(self, other):
        """Check for equality.

        Two DataCheckActionOption objs are considered equivalent if all of their attributes are equivalent.

        Args:
            other: An object to compare equality with.

        Returns:
            bool: True if the other object is considered an equivalent data check action, False otherwise.
        """
        attributes_to_check = [
            "action_code",
            "data_check_name",
            "parameters",
            "metadata",
        ]
        for attribute in attributes_to_check:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def to_dict(self):
        """Return a dictionary form of the data check action option."""
        action_option_dict = {
            "code": self.action_code.name,
            "data_check_name": self.data_check_name,
            "metadata": self.metadata,
            "parameters": self.parameters,
        }
        return action_option_dict

    @staticmethod
    def convert_dict_to_action(action_dict):
        """Convert a dictionary into a DataCheckActionOption.

        Args:
            action_dict: Dictionary to convert into action. Should have keys "code", "data_check_name", and "metadata".

        Raises:
            ValueError: If input dictionary does not have keys `code` and `metadata` and if the `metadata` dictionary does not have keys `columns` and `rows`.

        Returns:
            DataCheckActionOption object from the input dictionary.
        """
        if "code" not in action_dict or "metadata" not in action_dict:
            raise ValueError(
                "The input dictionary should have the keys `code` and `metadata`."
            )
        if (
            "columns" not in action_dict["metadata"]
            and "rows" not in action_dict["metadata"]
        ):
            raise ValueError(
                "The metadata dictionary should have the keys `columns` or `rows`. Set to None if not using."
            )

        return DataCheckActionOption(
            action_code=DataCheckActionCode._all_values[action_dict["code"]],
            metadata=action_dict["metadata"],
            data_check_name=action_dict["data_check_name"]
            if "data_check_name" in action_dict
            else None,
            parameters=action_dict["parameters"]
            if "parameters" in action_dict
            else None,
        )

    def _validate_parameters(self):
        """Validate parameters associated with the action option."""
        if self.parameters is None:
            return
        for _, parameter_value in self.parameters.items():
            if "parameter_type" not in parameter_value:
                raise ValueError("Each parameter must have a parameter_type key.")
            if parameter_value["parameter_type"] == "global":
                if "type" not in parameter_value:
                    raise ValueError("Each global parameter must have a type key.")
            elif parameter_value["parameter_type"] == "column":
                if "columns" not in parameter_value:
                    raise ValueError(
                        "Each `column` parameter type must also have a `columns` key indicating which columns the parameter should address."
                    )
                columns = parameter_value["columns"]
                if not isinstance(columns, dict):
                    raise ValueError(
                        "`columns` must be a dictionary, where each key is the name of a column and the associated value is a dictionary of parameters for that column."
                    )
                for column_parameters in columns.values():
                    for column_parameter_values in column_parameters.values():
                        if "type" not in column_parameter_values:
                            raise ValueError(
                                "Each column parameter must have a type key."
                            )
                        if "default_value" not in column_parameter_values:
                            raise ValueError(
                                "Each column parameter must have a default_value key."
                            )
