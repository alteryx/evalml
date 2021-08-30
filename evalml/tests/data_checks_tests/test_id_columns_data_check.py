import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckAction,
    DataCheckActionCode,
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
        ValueError, match="id_threshold must be a float between 0 and 1, inclusive."
    ):
        IDColumnsDataCheck(id_threshold=-0.1)
    with pytest.raises(
        ValueError, match="id_threshold must be a float between 0 and 1, inclusive."
    ):
        IDColumnsDataCheck(id_threshold=1.1)


def test_id_columns_warning():
    X_dict = {
        "col_1_id": [0, 1, 2, 3],
        "col_2": [2, 3, 4, 5],
        "col_3_id": [1, 1, 2, 3],
        "Id": [3, 1, 2, 0],
        "col_5": [0, 0, 1, 2],
        "col_6": [0.1, 0.2, 0.3, 0.4],
    }
    X = pd.DataFrame.from_dict(X_dict)
    id_cols_check = IDColumnsDataCheck(id_threshold=0.95)
    assert id_cols_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column 'Id' is 95.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "Id"},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'col_1_id' is 95.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "col_1_id"},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'col_2' is 95.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "col_2"},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'col_3_id' is 95.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "col_3_id"},
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "Id"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_1_id"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_2"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_3_id"}
            ).to_dict(),
        ],
    }

    X = pd.DataFrame.from_dict(X_dict)
    id_cols_check = IDColumnsDataCheck(id_threshold=1.0)
    assert id_cols_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column 'Id' is 100.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "Id"},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'col_1_id' is 100.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "col_1_id"},
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "Id"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_1_id"}
            ).to_dict(),
        ],
    }


def test_id_columns_strings():
    X_dict = {
        "col_1_id": ["a", "b", "c", "d"],
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
            "col_1_id": "categorical",
            "col_2": "categorical",
            "Id": "categorical",
            "col_5": "categorical",
        }
    )
    id_cols_check = IDColumnsDataCheck(id_threshold=0.95)
    assert id_cols_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column 'Id' is 95.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "Id"},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'col_1_id' is 95.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "col_1_id"},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'col_2' is 95.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "col_2"},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'col_3_id' is 95.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "col_3_id"},
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "Id"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_1_id"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_2"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_3_id"}
            ).to_dict(),
        ],
    }

    id_cols_check = IDColumnsDataCheck(id_threshold=1.0)
    assert id_cols_check.validate(X) == {
        "warnings": [
            DataCheckWarning(
                message="Column 'Id' is 100.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "Id"},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'col_1_id' is 100.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": "col_1_id"},
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "Id"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_1_id"}
            ).to_dict(),
        ],
    }


def test_id_cols_data_check_input_formats():
    id_cols_check = IDColumnsDataCheck(id_threshold=0.8)

    # test empty pd.DataFrame
    assert id_cols_check.validate(pd.DataFrame()) == {
        "warnings": [],
        "errors": [],
        "actions": [],
    }

    #  test Woodwork
    ww_input = pd.DataFrame(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]))
    ww_input.ww.init()
    assert id_cols_check.validate(ww_input) == {
        "warnings": [
            DataCheckWarning(
                message="Column '0' is 80.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": 0},
            ).to_dict(),
            DataCheckWarning(
                message="Column '1' is 80.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": 1},
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": 0}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": 1}
            ).to_dict(),
        ],
    }

    #  test 2D list
    assert id_cols_check.validate([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]) == {
        "warnings": [
            DataCheckWarning(
                message="Column '0' is 80.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": 0},
            ).to_dict(),
            DataCheckWarning(
                "Column '1' is 80.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": 1},
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": 0}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": 1}
            ).to_dict(),
        ],
    }

    # test np.array
    assert id_cols_check.validate(
        np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    ) == {
        "warnings": [
            DataCheckWarning(
                message="Column '0' is 80.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": 0},
            ).to_dict(),
            DataCheckWarning(
                message="Column '1' is 80.0% or more likely to be an ID column",
                data_check_name=id_data_check_name,
                message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                details={"column": 1},
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": 0}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": 1}
            ).to_dict(),
        ],
    }
