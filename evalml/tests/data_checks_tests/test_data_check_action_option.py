import re

import pytest

from evalml.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DCAOParameterType,
)
from evalml.data_checks.data_check_action import DataCheckAction


def test_data_check_action_option_attributes(dummy_data_check_name):
    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
    )
    assert data_check_action_option.data_check_name == dummy_data_check_name
    assert data_check_action_option.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action_option.metadata == {"rows": None, "columns": None}
    assert data_check_action_option.parameters == {}

    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        None,
        metadata={},
        parameters={},
    )
    assert data_check_action_option.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action_option.data_check_name is None
    assert data_check_action_option.metadata == {"rows": None, "columns": None}
    assert data_check_action_option.parameters == {}

    parameters = {
        "global_parameter_name": {
            "parameter_type": DCAOParameterType.GLOBAL,
            "type": "float",
            "default_value": 0.0,
        },
        "column_parameter_name": {
            "parameter_type": "column",
            "columns": {
                "a": {
                    "impute_strategy": {
                        "categories": ["mean", "most_frequent"],
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
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
    )
    data_check_action_option_eq = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
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
                "parameter_type": DCAOParameterType.GLOBAL,
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
                "parameter_type": DCAOParameterType.GLOBAL,
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
        DataCheckActionCode.DROP_COL,
        None,
        metadata={"metadata": ["this is different"]},
    )

    assert data_check_action_option != data_check_action_option_diff
    assert data_check_action_option_diff != data_check_action_option

    data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        data_check_name=None,
        metadata={"metadata": ["same metadata"]},
        parameters={
            "global_parameter_name": {
                "parameter_type": DCAOParameterType.GLOBAL,
                "type": "float",
                "default_value": 0.0,
            },
        },
    )
    data_check_action_option_diff_parameters = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        data_check_name=None,
        metadata={"metadata": ["same metadata"]},
        parameters={
            "different_global_parameter_name": {
                "parameter_type": DCAOParameterType.GLOBAL,
                "type": "float",
                "default_value": 0.0,
            },
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
                "parameter_type": DCAOParameterType.GLOBAL.value,
                "type": "float",
                "default_value": 0.0,
            },
        },
    )

    assert data_check_action_option.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "parameters": {},
        "metadata": {"columns": None, "rows": None},
    }
    assert data_check_action_option_empty_metadata.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "parameters": {},
        "metadata": {"columns": None, "rows": None},
    }
    assert data_check_action_option_with_metadata.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "parameters": {},
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
                "parameter_type": DCAOParameterType.GLOBAL.value,
                "type": "float",
                "default_value": 0.0,
            },
        },
    }


def test_convert_dict_to_option_bad_input():
    data_check_action_option_dict_no_code = {
        "metadata": {"columns": None, "rows": None},
    }
    with pytest.raises(ValueError, match="The input dictionary should have the keys"):
        DataCheckActionOption.convert_dict_to_option(
            data_check_action_option_dict_no_code,
        )

    data_check_action_option_dict_no_metadata = {
        "code": DataCheckActionCode.DROP_COL.name,
    }
    with pytest.raises(ValueError, match="The input dictionary should have the keys"):
        DataCheckActionOption.convert_dict_to_option(
            data_check_action_option_dict_no_metadata,
        )

    data_check_action_option_dict_no_columns = {
        "code": DataCheckActionCode.DROP_COL.name,
        "metadata": {"cow": None},
    }
    with pytest.raises(
        ValueError,
        match="The metadata dictionary should have the keys",
    ):
        DataCheckActionOption.convert_dict_to_option(
            data_check_action_option_dict_no_columns,
        )


def test_convert_dict_to_option_bad_parameter_input(dummy_data_check_name):
    with pytest.raises(
        ValueError,
        match="Each parameter must have a parameter_type key.",
    ):
        DataCheckActionOption(
            action_code=DataCheckActionCode.DROP_COL,
            data_check_name=dummy_data_check_name,
            metadata={"columns": None, "rows": None},
            parameters={
                "global_parameter_name": {
                    "type": "float",
                    "default_value": 0.0,
                },
            },
        )
    with pytest.raises(
        ValueError,
        match="Each parameter must have a parameter_type key with a value of `global` or `column`.",
    ):
        DataCheckActionOption(
            action_code=DataCheckActionCode.DROP_COL,
            data_check_name=dummy_data_check_name,
            metadata={"columns": None, "rows": None},
            parameters={
                "global_parameter_name": {
                    "parameter_type": "invalid_parameter_type",
                    "type": "float",
                    "default_value": 0.0,
                },
            },
        )

    with pytest.raises(ValueError, match="Each global parameter must have a type key."):
        DataCheckActionOption(
            action_code=DataCheckActionCode.DROP_COL,
            data_check_name=dummy_data_check_name,
            metadata={"columns": None, "rows": None},
            parameters={
                "global_parameter_name": {
                    "parameter_type": DCAOParameterType.GLOBAL,
                    "default_value": 0.0,
                },
            },
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Each `column` parameter type must also have a `columns` key indicating which columns the parameter should address",
        ),
    ):
        DataCheckActionOption(
            action_code=DataCheckActionCode.DROP_COL,
            data_check_name=dummy_data_check_name,
            metadata={"columns": None, "rows": None},
            parameters={
                "columns_parameter_name": {
                    "parameter_type": "column",
                },
            },
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "`columns` must be a dictionary, where each key is the name of a column and the associated value is a dictionary of parameters for that column",
        ),
    ):
        DataCheckActionOption(
            action_code=DataCheckActionCode.DROP_COL,
            data_check_name=dummy_data_check_name,
            metadata={"columns": None, "rows": None},
            parameters={
                "columns_parameter_name": {
                    "parameter_type": "column",
                    "columns": "some incorrect string input",
                },
            },
        )
    with pytest.raises(ValueError, match="Each column parameter must have a type key."):
        DataCheckActionOption(
            action_code=DataCheckActionCode.DROP_COL,
            data_check_name=dummy_data_check_name,
            metadata={"columns": None, "rows": None},
            parameters={
                "columns_parameter_name": {
                    "parameter_type": "column",
                    "columns": {
                        "some_column_name": {
                            "per_column_parameter": {
                                "default_value": 0.0,
                            },
                        },
                    },
                },
            },
        )
    with pytest.raises(
        ValueError,
        match="Each column parameter must have a default_value key.",
    ):
        DataCheckActionOption(
            action_code=DataCheckActionCode.DROP_COL,
            data_check_name=dummy_data_check_name,
            metadata={"columns": None, "rows": None},
            parameters={
                "columns_parameter_name": {
                    "parameter_type": "column",
                    "columns": {
                        "some_column_name": {"per_column_parameter": {"type": "float"}},
                    },
                },
            },
        )


def test_convert_dict_to_option(dummy_data_check_name):
    data_check_action_option_dict = {
        "code": DataCheckActionCode.DROP_COL.name,
        "metadata": {"columns": None, "rows": None},
    }
    expected_data_check_action_option = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        None,
    )
    data_check_action_option = DataCheckActionOption.convert_dict_to_option(
        data_check_action_option_dict,
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
    data_check_action_option = DataCheckActionOption.convert_dict_to_option(
        data_check_action_option_dict_with_other_metadata,
    )
    assert data_check_action_option == expected_data_check_action_option


def test_get_action_from_defaults(dummy_data_check_name):
    data_check_action_option_with_no_parameters = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={"columns": None, "rows": None},
        parameters={},
    )
    assert (
        data_check_action_option_with_no_parameters.get_action_from_defaults()
        == DataCheckAction(
            DataCheckActionCode.DROP_COL.name,
            dummy_data_check_name,
            metadata={"columns": None, "rows": None, "parameters": {}},
        )
    )

    data_check_action_option_with_one_column_parameter = DataCheckActionOption(
        DataCheckActionCode.IMPUTE_COL,
        dummy_data_check_name,
        metadata={"columns": None, "rows": None},
        parameters={
            "impute_strategies": {
                "parameter_type": "column",
                "columns": {
                    "some_column": {
                        "impute_strategy": {
                            "categories": ["mean", "most_frequent"],
                            "type": "category",
                            "default_value": "most_frequent",
                        },
                        "fill_value": {"type": "float", "default_value": 0.0},
                    },
                    "some_other_column": {
                        "impute_strategy": {
                            "categories": ["mean", "most_frequent"],
                            "type": "category",
                            "default_value": "mean",
                        },
                        "fill_value": {"type": "float", "default_value": 1.0},
                    },
                },
            },
        },
    )
    assert (
        data_check_action_option_with_one_column_parameter.get_action_from_defaults()
        == DataCheckAction(
            DataCheckActionCode.IMPUTE_COL.name,
            dummy_data_check_name,
            metadata={
                "columns": None,
                "rows": None,
                "parameters": {
                    "impute_strategies": {
                        "some_column": {
                            "impute_strategy": "most_frequent",
                            "fill_value": 0.0,
                        },
                        "some_other_column": {
                            "impute_strategy": "mean",
                            "fill_value": 1.0,
                        },
                    },
                },
            },
        )
    )

    data_check_action_option_with_global_parameter = DataCheckActionOption(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        parameters={
            "global_parameter_name": {
                "parameter_type": DCAOParameterType.GLOBAL,
                "type": "float",
                "default_value": 0.0,
            },
        },
    )
    assert (
        data_check_action_option_with_global_parameter.get_action_from_defaults()
        == DataCheckAction(
            DataCheckActionCode.DROP_COL.name,
            dummy_data_check_name,
            metadata={
                "columns": None,
                "rows": None,
                "parameters": {"global_parameter_name": 0.0},
            },
        )
    )

    data_check_action_option_with_multiple_parameters = DataCheckActionOption(
        DataCheckActionCode.IMPUTE_COL,
        dummy_data_check_name,
        parameters={
            "global_parameter_name": {
                "parameter_type": DCAOParameterType.GLOBAL,
                "type": "float",
                "default_value": 0.0,
            },
            "impute_strategies": {
                "parameter_type": "column",
                "columns": {
                    "some_column": {
                        "impute_strategy": {
                            "categories": ["mean", "most_frequent"],
                            "type": "category",
                            "default_value": "most_frequent",
                        },
                        "fill_value": {"type": "float", "default_value": 0.0},
                    },
                    "some_other_column": {
                        "impute_strategy": {
                            "categories": ["mean", "most_frequent"],
                            "type": "category",
                            "default_value": "mean",
                        },
                        "fill_value": {"type": "float", "default_value": 1.0},
                    },
                },
            },
        },
    )
    assert (
        data_check_action_option_with_multiple_parameters.get_action_from_defaults()
        == DataCheckAction(
            DataCheckActionCode.IMPUTE_COL.name,
            dummy_data_check_name,
            metadata={
                "columns": None,
                "rows": None,
                "parameters": {
                    "global_parameter_name": 0.0,
                    "impute_strategies": {
                        "some_column": {
                            "impute_strategy": "most_frequent",
                            "fill_value": 0.0,
                        },
                        "some_other_column": {
                            "impute_strategy": "mean",
                            "fill_value": 1.0,
                        },
                    },
                },
            },
        )
    )


def test_dcao_parameter_type_to_str():
    assert str(DCAOParameterType.GLOBAL) == "global"
    assert str(DCAOParameterType.COLUMN) == "column"


def test_handle_dcao_parameter_type():
    DCAOParameterType.handle_dcao_parameter_type("global") == DCAOParameterType.GLOBAL
    DCAOParameterType.handle_dcao_parameter_type("column") == DCAOParameterType.COLUMN

    DCAOParameterType.handle_dcao_parameter_type("GLOBAL") == DCAOParameterType.GLOBAL
    DCAOParameterType.handle_dcao_parameter_type("COLUMN") == DCAOParameterType.COLUMN

    DCAOParameterType.handle_dcao_parameter_type(
        DCAOParameterType.GLOBAL,
    ) == DCAOParameterType.GLOBAL
    DCAOParameterType.handle_dcao_parameter_type(
        DCAOParameterType.COLUMN,
    ) == DCAOParameterType.COLUMN


def test_handle_dcao_parameter_type_invalid():
    with pytest.raises(
        ValueError,
        match="`handle_dcao_parameter_type` was not passed a str or DCAOParameterType object",
    ):
        DCAOParameterType.handle_dcao_parameter_type(None)
