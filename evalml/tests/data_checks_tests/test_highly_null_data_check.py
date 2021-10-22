import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning,
    HighlyNullDataCheck,
)

highly_null_data_check_name = HighlyNullDataCheck.name


class SeriesWrap:
    def __init__(self, series):
        self.series = series

    def __eq__(self, series_2):
        return all(self.series.eq(series_2.series))


def test_highly_null_data_check_init():
    highly_null_check = HighlyNullDataCheck()
    assert highly_null_check.pct_null_col_threshold == 0.95
    assert highly_null_check.pct_null_row_threshold == 0.95

    highly_null_check = HighlyNullDataCheck(pct_null_col_threshold=0.0)
    assert highly_null_check.pct_null_col_threshold == 0
    assert highly_null_check.pct_null_row_threshold == 0.95

    highly_null_check = HighlyNullDataCheck(pct_null_row_threshold=0.5)
    assert highly_null_check.pct_null_col_threshold == 0.95
    assert highly_null_check.pct_null_row_threshold == 0.5

    highly_null_check = HighlyNullDataCheck(
        pct_null_col_threshold=1.0, pct_null_row_threshold=1.0
    )
    assert highly_null_check.pct_null_col_threshold == 1.0
    assert highly_null_check.pct_null_row_threshold == 1.0

    with pytest.raises(
        ValueError,
        match="pct null column threshold must be a float between 0 and 1, inclusive.",
    ):
        HighlyNullDataCheck(pct_null_col_threshold=-0.1)
    with pytest.raises(
        ValueError,
        match="pct null column threshold must be a float between 0 and 1, inclusive.",
    ):
        HighlyNullDataCheck(pct_null_col_threshold=1.1)
    with pytest.raises(
        ValueError,
        match="pct null row threshold must be a float between 0 and 1, inclusive.",
    ):
        HighlyNullDataCheck(pct_null_row_threshold=-0.5)
    with pytest.raises(
        ValueError,
        match="pct null row threshold must be a float between 0 and 1, inclusive.",
    ):
        HighlyNullDataCheck(pct_null_row_threshold=2.1)


def test_highly_null_data_check_warnings():
    data = pd.DataFrame(
        {
            "lots_of_null": [None, None, None, None, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        }
    )
    no_null_check = HighlyNullDataCheck(
        pct_null_col_threshold=0.0, pct_null_row_threshold=0.0
    )
    highly_null_rows = SeriesWrap(pd.Series([2 / 3, 2 / 3, 2 / 3, 2 / 3, 1 / 3]))
    validate_results = no_null_check.validate(data)
    validate_results["warnings"][0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_results["warnings"][0]["details"]["pct_null_cols"]
    )
    assert validate_results == {
        "warnings": [
            DataCheckWarning(
                message="5 out of 5 rows are more than 0.0% null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                details={
                    "pct_null_cols": highly_null_rows,
                    "rows": highly_null_rows.series.index.tolist(),
                },
            ).to_dict(),
            DataCheckWarning(
                message="Columns 'lots_of_null', 'all_null' are 0.0% or more null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={
                    "columns": ["lots_of_null", "all_null"],
                    "pct_null_rows": {"all_null": 1.0, "lots_of_null": 0.8},
                    "null_row_indices": {
                        "all_null": [0, 1, 2, 3, 4],
                        "lots_of_null": [0, 1, 2, 3],
                    },
                },
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_ROWS, metadata={"rows": [0, 1, 2, 3, 4]}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL,
                metadata={"columns": ["lots_of_null", "all_null"]},
            ).to_dict(),
        ],
    }

    some_null_check = HighlyNullDataCheck(
        pct_null_col_threshold=0.5, pct_null_row_threshold=0.5
    )
    highly_null_rows = SeriesWrap(pd.Series([2 / 3, 2 / 3, 2 / 3, 2 / 3]))
    validate_results = some_null_check.validate(data)
    validate_results["warnings"][0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_results["warnings"][0]["details"]["pct_null_cols"]
    )
    assert validate_results == {
        "warnings": [
            DataCheckWarning(
                message="4 out of 5 rows are more than 50.0% null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                details={"pct_null_cols": highly_null_rows, "rows": [0, 1, 2, 3]},
            ).to_dict(),
            DataCheckWarning(
                message="Columns 'lots_of_null', 'all_null' are 50.0% or more null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={
                    "columns": ["lots_of_null", "all_null"],
                    "pct_null_rows": {"all_null": 1.0, "lots_of_null": 0.8},
                    "null_row_indices": {
                        "all_null": [0, 1, 2, 3, 4],
                        "lots_of_null": [0, 1, 2, 3],
                    },
                },
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_ROWS, metadata={"rows": [0, 1, 2, 3]}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL,
                metadata={"columns": ["lots_of_null", "all_null"]},
            ).to_dict(),
        ],
    }

    all_null_check = HighlyNullDataCheck(
        pct_null_col_threshold=1.0, pct_null_row_threshold=1.0
    )
    assert all_null_check.validate(data) == {
        "warnings": [
            DataCheckWarning(
                message="Columns 'all_null' are 100.0% or more null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={
                    "columns": ["all_null"],
                    "pct_null_rows": {"all_null": 1.0},
                    "null_row_indices": {"all_null": [0, 1, 2, 3, 4]},
                },
            ).to_dict()
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"columns": ["all_null"]}
            ).to_dict()
        ],
    }


def test_highly_null_data_check_separate_rows_cols():
    data = pd.DataFrame(
        {
            "lots_of_null": [None, None, None, None, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        }
    )
    row_null_check = HighlyNullDataCheck(
        pct_null_col_threshold=0.9, pct_null_row_threshold=0.0
    )
    highly_null_rows = SeriesWrap(pd.Series([2 / 3, 2 / 3, 2 / 3, 2 / 3, 1 / 3]))
    validate_results = row_null_check.validate(data)
    validate_results["warnings"][0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_results["warnings"][0]["details"]["pct_null_cols"]
    )
    assert validate_results == {
        "warnings": [
            DataCheckWarning(
                message="5 out of 5 rows are more than 0.0% null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                details={"pct_null_cols": highly_null_rows, "rows": [0, 1, 2, 3, 4]},
            ).to_dict(),
            DataCheckWarning(
                message="Columns 'all_null' are 90.0% or more null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={
                    "columns": ["all_null"],
                    "pct_null_rows": {"all_null": 1.0},
                    "null_row_indices": {"all_null": [0, 1, 2, 3, 4]},
                },
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_ROWS, metadata={"rows": [0, 1, 2, 3, 4]}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"columns": ["all_null"]}
            ).to_dict(),
        ],
    }

    col_null_check = HighlyNullDataCheck(
        pct_null_col_threshold=0.0, pct_null_row_threshold=0.9
    )
    validate_results = col_null_check.validate(data)
    assert validate_results == {
        "warnings": [
            DataCheckWarning(
                message="Columns 'lots_of_null', 'all_null' are 0.0% or more null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={
                    "columns": ["lots_of_null", "all_null"],
                    "pct_null_rows": {"lots_of_null": 0.8, "all_null": 1.0},
                    "null_row_indices": {
                        "all_null": [0, 1, 2, 3, 4],
                        "lots_of_null": [0, 1, 2, 3],
                    },
                },
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL,
                metadata={"columns": ["lots_of_null", "all_null"]},
            ).to_dict(),
        ],
    }


def test_highly_null_data_check_input_formats():
    highly_null_check = HighlyNullDataCheck(
        pct_null_col_threshold=0.8, pct_null_row_threshold=0.8
    )

    # test empty pd.DataFrame
    assert highly_null_check.validate(pd.DataFrame()) == {
        "warnings": [],
        "errors": [],
        "actions": [],
    }

    highly_null_rows = SeriesWrap(pd.Series([0.8]))
    expected = {
        "warnings": [
            DataCheckWarning(
                message="1 out of 2 rows are more than 80.0% null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                details={"pct_null_cols": highly_null_rows, "rows": [0]},
            ).to_dict(),
            DataCheckWarning(
                message="Columns '0', '1', '2' are 80.0% or more null",
                data_check_name=highly_null_data_check_name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={
                    "columns": [0, 1, 2],
                    "pct_null_rows": {0: 1.0, 1: 1.0, 2: 1.0},
                    "null_row_indices": {0: [0, 1], 1: [0, 1], 2: [0, 1]},
                },
            ).to_dict(),
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_ROWS, metadata={"rows": [0]}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"columns": [0, 1, 2]}
            ).to_dict(),
        ],
    }
    #  test Woodwork
    ww_input = pd.DataFrame([[None, None, None, None, 0], [None, None, None, "hi", 5]])
    ww_input.ww.init()
    validate_results = highly_null_check.validate(ww_input)
    validate_results["warnings"][0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_results["warnings"][0]["details"]["pct_null_cols"]
    )
    assert validate_results == expected

    # #  test 2D list
    validate_results = highly_null_check.validate(
        [[None, None, None, None, 0], [None, None, None, "hi", 5]]
    )
    validate_results["warnings"][0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_results["warnings"][0]["details"]["pct_null_cols"]
    )
    assert validate_results == expected

    # test np.array
    validate_results = highly_null_check.validate(
        np.array([[None, None, None, None, 0], [None, None, None, "hi", 5]])
    )
    validate_results["warnings"][0]["details"]["pct_null_cols"] = SeriesWrap(
        validate_results["warnings"][0]["details"]["pct_null_cols"]
    )
    assert validate_results == expected
