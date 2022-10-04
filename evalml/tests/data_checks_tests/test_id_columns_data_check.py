import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
    IDColumnsDataCheck,
)

id_data_check_name = IDColumnsDataCheck.name


def test_id_cols_data_check_init():
    id_cols_check = IDColumnsDataCheck()
    assert id_cols_check.id_threshold == 1.0

    id_cols_check = IDColumnsDataCheck(id_threshold=0.0)
    assert id_cols_check.id_threshold == 0

    id_cols_check = IDColumnsDataCheck(id_threshold=0.5)
    assert id_cols_check.id_threshold == 0.5

    id_cols_check = IDColumnsDataCheck(id_threshold=1.0)
    assert id_cols_check.id_threshold == 1.0

    with pytest.raises(
        ValueError,
        match="id_threshold must be a float between 0 and 1, inclusive.",
    ):
        IDColumnsDataCheck(id_threshold=-0.1)
    with pytest.raises(
        ValueError,
        match="id_threshold must be a float between 0 and 1, inclusive.",
    ):
        IDColumnsDataCheck(id_threshold=1.1)


def test_id_columns_warning():
    X_dict = {
        "col_1": [1, 1, 2, 3],
        "col_2": [2, 3, 4, 5],
        "col_3_id": [0, 1, 2, 3],
        "Id": [3, 1, 2, 0],
        "col_5": [0, 0, 1, 2],
        "col_6": [0.1, 0.2, 0.3, 0.4],
    }
    X = pd.DataFrame.from_dict(X_dict)
    id_cols_check = IDColumnsDataCheck(id_threshold=0.95)
    assert id_cols_check.validate(X) == [
        DataCheckWarning(
            message="Columns 'Id', 'col_2', 'col_3_id' are 95.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["Id", "col_2", "col_3_id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["Id", "col_2", "col_3_id"]},
                ),
            ],
        ).to_dict(),
    ]

    X = pd.DataFrame.from_dict(X_dict)
    id_cols_check = IDColumnsDataCheck(id_threshold=1.0)
    assert id_cols_check.validate(X) == [
        DataCheckWarning(
            message="Columns 'Id', 'col_3_id' are 100.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["Id", "col_3_id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["Id", "col_3_id"]},
                ),
            ],
        ).to_dict(),
    ]


def test_id_columns_strings():
    X_dict = {
        "col_1": ["b", "b", "c", "d"],
        "col_2": ["w", "x", "y", "z"],
        "col_3_id": [
            "123456789012345",
            "234567890123456",
            "3456789012345678",
            "45678901234567",
        ],
        "Id": ["z", "y", "x", "a"],
        "col_5": ["0", "0", "1", "2"],
        "col_6": [0.1, 0.2, 0.3, 0.4],
    }
    X = pd.DataFrame.from_dict(X_dict)
    X.ww.init(
        logical_types={
            "col_1": "categorical",
            "col_2": "categorical",
            "Id": "categorical",
            "col_5": "categorical",
        },
    )
    id_cols_check = IDColumnsDataCheck(id_threshold=0.95)
    assert id_cols_check.validate(X) == [
        DataCheckWarning(
            message="Columns 'Id', 'col_2', 'col_3_id' are 95.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["Id", "col_2", "col_3_id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["Id", "col_2", "col_3_id"]},
                ),
            ],
        ).to_dict(),
    ]

    id_cols_check = IDColumnsDataCheck(id_threshold=1.0)
    assert id_cols_check.validate(X) == [
        DataCheckWarning(
            message="Columns 'Id' are 100.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["Id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["Id"]},
                ),
            ],
        ).to_dict(),
    ]


@pytest.mark.parametrize("logical_type", ["integer", "integer_nullable", "double"])
def test_id_cols_data_check_input_formats(logical_type):
    id_cols_check = IDColumnsDataCheck(id_threshold=0.8)

    # test empty pd.DataFrame
    assert id_cols_check.validate(pd.DataFrame()) == []

    #  test Woodwork
    ww_input = pd.DataFrame(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]))
    logical_types = {col: logical_type for col in ww_input.columns}
    ww_input.ww.init(logical_types=logical_types)
    assert id_cols_check.validate(ww_input) == [
        DataCheckWarning(
            message="Columns '0', '1' are 80.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": [0, 1]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": [0, 1]},
                ),
            ],
        ).to_dict(),
    ]

    #  test 2D list
    assert id_cols_check.validate([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]) == [
        DataCheckWarning(
            message="Columns '0', '1' are 80.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": [0, 1]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": [0, 1]},
                ),
            ],
        ).to_dict(),
    ]

    # test np.array
    assert id_cols_check.validate(
        np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
    ) == [
        DataCheckWarning(
            message="Columns '0', '1' are 80.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": [0, 1]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": [0, 1]},
                ),
            ],
        ).to_dict(),
    ]


@pytest.mark.parametrize(
    "input_type", ["integer", "integer_nullable", "string", "double"]
)
def test_identified_first_col_primary_key(
    input_type, get_test_data_with_or_without_primary_key
):
    X = get_test_data_with_or_without_primary_key(
        input_type=input_type, has_primary_key=True
    )
    id_cols_check = IDColumnsDataCheck(id_threshold=0.95)
    assert id_cols_check.validate(X) == [
        DataCheckWarning(
            message="The first column 'col_1_id' is likely to be the primary key",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_FIRST_COLUMN,
            details={"columns": ["col_1_id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.SET_FIRST_COL_ID,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["col_1_id"]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Columns 'col_2', 'col_3_id' are 95.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["col_2", "col_3_id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["col_2", "col_3_id"]},
                ),
            ],
        ).to_dict(),
    ]

    X = X.rename(columns={"col_1_id": "ID"})
    assert id_cols_check.validate(X) == [
        DataCheckWarning(
            message="The first column 'ID' is likely to be the primary key",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_FIRST_COLUMN,
            details={"columns": ["ID"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.SET_FIRST_COL_ID,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["ID"]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Columns 'col_2', 'col_3_id' are 95.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["col_2", "col_3_id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["col_2", "col_3_id"]},
                ),
            ],
        ).to_dict(),
    ]


@pytest.mark.parametrize(
    "input_type", ["integer", "integer_nullable", "string", "double"]
)
def test_unidentified_first_col_primary_key(
    input_type, get_test_data_with_or_without_primary_key
):
    X = get_test_data_with_or_without_primary_key(
        input_type=input_type, has_primary_key=False
    )

    id_cols_check = IDColumnsDataCheck(id_threshold=0.95)
    assert id_cols_check.validate(X) == [
        DataCheckWarning(
            message="Columns 'col_2', 'col_1_id', 'col_3_id' are 95.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["col_2", "col_1_id", "col_3_id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["col_2", "col_1_id", "col_3_id"]},
                ),
            ],
        ).to_dict(),
    ]

    X = X.rename(columns={"col_1_id": "col_1"})
    if input_type == "integer":
        X.at[0, "col_1"] = 0
    elif input_type == "integer_nullable":
        X.at[0, "col_1"] = 0
    elif input_type == "double":
        X.at[0, "col_1"] = 0.0
    elif input_type == "string":
        X["col_1"] = X["col_1"].cat.add_categories("a")
        X.at[0, "col_1"] = "a"

    assert id_cols_check.validate(X) == [
        DataCheckWarning(
            message="Columns 'col_1', 'col_2', 'col_3_id' are 95.0% or more likely to be an ID column",
            data_check_name=id_data_check_name,
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["col_1", "col_2", "col_3_id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name=id_data_check_name,
                    metadata={"columns": ["col_1", "col_2", "col_3_id"]},
                ),
            ],
        ).to_dict(),
    ]
