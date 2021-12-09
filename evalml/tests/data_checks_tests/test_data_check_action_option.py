import pytest

from evalml.data_checks import DataCheckActionCode, DataCheckActionOption


@pytest.fixture
def dummy_data_check_name():
    return "dummy_data_check_name"


def test_data_check_action_option_attributes(dummy_data_check_name):
    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL, dummy_data_check_name
    )
    assert data_check_action_option.data_check_name == dummy_data_check_name
    assert data_check_action_option.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action_option.metadata == {"rows": None, "columns": None}
    assert data_check_action_option.parameters == None

    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL, None, metadata={}, parameters={}
    )
    assert data_check_action_option.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action_option.data_check_name is None
    assert data_check_action_option.metadata == {"rows": None, "columns": None}
    assert data_check_action_option.parameters == {}

    parameters = {
        "global_parameter_name": {
            "parameter_type": "global",
            "type": "float",
            "default_value": 0.0,
        },
        "column_parameter_name": {
            "parameter_type": "column",
            "columns": {
                "a": {
                    "impute_strategy": {
                        "categories": ["mean", "mode"],
                        "type": "category",
                        "default_value": "mean",
                    },
                    "constant_fill_value": {"type": "float", "default_value": 0},
                },
            },
        },
    }
    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={"columns": [1, 2]},
        parameters=parameters,
    )
    assert data_check_action_option.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action_option.data_check_name == dummy_data_check_name
    assert data_check_action_option.metadata == {"columns": [1, 2], "rows": None}
    assert data_check_action_option.parameters == parameters


def test_data_check_action_option_equality(dummy_data_check_name):
    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL, dummy_data_check_name
    )
    data_check_action_option_eq = DataCheckActionOption(
        DataCheckActionCode.DROP_COL, dummy_data_check_name
    )
    assert data_check_action_option == data_check_action_option
    assert data_check_action_option == data_check_action_option_eq
    assert data_check_action_option_eq == data_check_action_option

    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        None,
        metadata={"same detail": "same same same"},
        parameters={
            "global_parameter_name": {
                "parameter_type": "global",
                "type": "float",
                "default_value": 0.0,
            },
        },
    )
    data_check_action_option_eq = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        None,
        metadata={"same detail": "same same same"},
        parameters={
            "global_parameter_name": {
                "parameter_type": "global",
                "type": "float",
                "default_value": 0.0,
            },
        },
    )
    assert data_check_action_option == data_check_action_option
    assert data_check_action_option == data_check_action_option_eq
    assert data_check_action_option_eq == data_check_action_option


def test_data_check_action_option_inequality():
    data_check_action_option = DataCheckActionOption(DataCheckActionCode.DROP_COL, None)
    data_check_action_option_diff = DataCheckActionOption(
        DataCheckActionCode.DROP_COL, None, metadata={"metadata": ["this is different"]}
    )

    assert data_check_action_option != data_check_action_option_diff
    assert data_check_action_option_diff != data_check_action_option

    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        data_check_name=None,
        metadata={"metadata": ["same metadata"]},
        parameters={
            "global_parameter_name": {
                "parameter_type": "global",
                "type": "float",
                "default_value": 0.0,
            }
        },
    )
    data_check_action_option_diff_parameters = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        data_check_name=None,
        metadata={"metadata": ["same metadata"]},
        parameters={
            "different_global_parameter_name": {
                "parameter_type": "global",
                "type": "float",
                "default_value": 0.0,
            }
        },
    )

    assert data_check_action_option != data_check_action_option_diff
    assert data_check_action_option_diff != data_check_action_option
    assert data_check_action_option != data_check_action_option_diff_parameters
    assert data_check_action_option_diff_parameters != data_check_action_option


def test_data_check_action_option_to_dict(dummy_data_check_name):
    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
    )
    data_check_action_option_empty_metadata = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={},
    )
    data_check_action_option_with_metadata = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={"some detail": ["some detail value"]},
    )
    data_check_action_option_with_parameters = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={"some detail": ["some detail value"]},
        parameters={
            "global_parameter_name": {
                "parameter_type": "global",
                "type": "float",
                "default_value": 0.0,
            }
        },
    )

    assert data_check_action_option.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "parameters": None,
        "metadata": {"columns": None, "rows": None},
    }
    assert data_check_action_option_empty_metadata.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "parameters": None,
        "metadata": {"columns": None, "rows": None},
    }
    assert data_check_action_option_with_metadata.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "parameters": None,
        "metadata": {
            "some detail": ["some detail value"],
            "columns": None,
            "rows": None,
        },
    }
    assert data_check_action_option_with_parameters.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "metadata": {
            "some detail": ["some detail value"],
            "columns": None,
            "rows": None,
        },
        "parameters": {
            "global_parameter_name": {
                "parameter_type": "global",
                "type": "float",
                "default_value": 0.0,
            }
        },
    }


def test_convert_dict_to_action_bad_input():
    data_check_action_option_dict_no_code = {
        "metadata": {"columns": None, "rows": None},
    }
    with pytest.raises(ValueError, match="The input dictionary should have the keys"):
        DataCheckActionOption.convert_dict_to_action(
            data_check_action_option_dict_no_code
        )

    data_check_action_option_dict_no_metadata = {
        "code": DataCheckActionCode.DROP_COL.name,
    }
    with pytest.raises(ValueError, match="The input dictionary should have the keys"):
        DataCheckActionOption.convert_dict_to_action(
            data_check_action_option_dict_no_metadata
        )

    data_check_action_option_dict_no_columns = {
        "code": DataCheckActionCode.DROP_COL.name,
        "metadata": {"rows": None},
    }
    with pytest.raises(
        ValueError, match="The metadata dictionary should have the keys"
    ):
        DataCheckActionOption.convert_dict_to_action(
            data_check_action_option_dict_no_columns
        )

    data_check_action_option_dict_no_rows = {
        "code": DataCheckActionCode.DROP_COL.name,
        "metadata": {"columns": None},
    }
    with pytest.raises(
        ValueError, match="The metadata dictionary should have the keys"
    ):
        DataCheckActionOption.convert_dict_to_action(
            data_check_action_option_dict_no_rows
        )


def test_convert_dict_to_action_bad_parameter_input(dummy_data_check_name):
    with pytest.raises(
        ValueError, match="Each parameter must have a parameter_type key."
    ):
        DataCheckActionOption(
            action_code=DataCheckActionCode.DROP_COL,
            data_check_name=dummy_data_check_name,
            metadata={"columns": None, "rows": None},
            parameters={
                "global_parameter_name": {
                    "type": "float",
                    "default_value": 0.0,
                }
            },
        )


def test_convert_dict_to_action(dummy_data_check_name):
    data_check_action_option_dict = {
        "code": DataCheckActionCode.DROP_COL.name,
        "metadata": {"columns": None, "rows": None},
    }
    expected_data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL, None
    )
    data_check_action_option = DataCheckActionOption.convert_dict_to_action(
        data_check_action_option_dict
    )
    assert data_check_action_option == expected_data_check_action_option

    data_check_action_option_dict_with_other_metadata = {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "parameters": None,
        "metadata": {
            "some detail": ["this is different"],
            "columns": None,
            "rows": None,
        },
    }
    expected_data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={"some detail": ["this is different"]},
    )
    data_check_action_option = DataCheckActionOption.convert_dict_to_action(
        data_check_action_option_dict_with_other_metadata
    )
    assert data_check_action_option == expected_data_check_action_option
