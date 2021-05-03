import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import (
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning,
    HighlyNullRowsDataCheck
)


def test_highly_null_data_check_init():
    highly_null_check = HighlyNullRowsDataCheck()
    assert highly_null_check.pct_null_threshold == 0.95

    highly_null_check = HighlyNullRowsDataCheck(pct_null_threshold=0.0)
    assert highly_null_check.pct_null_threshold == 0

    highly_null_check = HighlyNullRowsDataCheck(pct_null_threshold=0.5)
    assert highly_null_check.pct_null_threshold == 0.5

    highly_null_check = HighlyNullRowsDataCheck(pct_null_threshold=1.0)
    assert highly_null_check.pct_null_threshold == 1.0

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        HighlyNullRowsDataCheck(pct_null_threshold=-0.1)
    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        HighlyNullRowsDataCheck(pct_null_threshold=1.1)


def test_highly_null_data_check_warnings():
    data = pd.DataFrame({'a': [None, None, 10],
                         'b': [None, "text", "text_1"]})
    zero_null_check = HighlyNullRowsDataCheck(pct_null_threshold=0.0)
    assert zero_null_check.validate(data) == {
        'warnings': [DataCheckWarning(message="Row '0' is more than 0% null",
                                      data_check_name=HighlyNullRowsDataCheck.name,
                                      message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                      details={'row': 0, 'pct_null_cols': 1.0}).to_dict(),
                     DataCheckWarning(message="Row '1' is more than 0% null",
                                      data_check_name=HighlyNullRowsDataCheck.name,
                                      message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                      details={'row': 1, 'pct_null_cols': 0.5}).to_dict()],
        'errors': [],
        'actions': [DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 0}).to_dict(),
                    DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 1}).to_dict()]}

    fifty_null_check = HighlyNullRowsDataCheck(pct_null_threshold=0.5)
    assert fifty_null_check.validate(data) == {
        'warnings': [DataCheckWarning(message="Row '0' is 50.0% or more null",
                                      data_check_name=HighlyNullRowsDataCheck.name,
                                      message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                      details={'row': 0, 'pct_null_cols': 1.0}).to_dict(),
                     DataCheckWarning(message="Row '1' is 50.0% or more null",
                                      data_check_name=HighlyNullRowsDataCheck.name,
                                      message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                      details={'row': 1, 'pct_null_cols': 0.5}).to_dict()],
        'errors': [],
        'actions': [DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 0}).to_dict(),
                    DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 1}).to_dict()]}

    hundred_null_check = HighlyNullRowsDataCheck(pct_null_threshold=1.0)
    assert hundred_null_check.validate(data) == {
        'warnings': [DataCheckWarning(message="Row '0' is 100.0% or more null",
                                      data_check_name=HighlyNullRowsDataCheck.name,
                                      message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                      details={'row': 0, 'pct_null_cols': 1.0}).to_dict()],
        'errors': [],
        'actions': [DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 0}).to_dict()]}


def test_highly_null_data_check_input_formats():
    highly_null_rows_check = HighlyNullRowsDataCheck(pct_null_threshold=0.5)

    # test empty pd.DataFrame
    assert highly_null_rows_check.validate(pd.DataFrame()) == {"warnings": [], "errors": [], "actions": []}

    expected = {
        'warnings': [DataCheckWarning(message="Row '0' is 50.0% or more null",
                                      data_check_name=HighlyNullRowsDataCheck.name,
                                      message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                      details={'row': 0, 'pct_null_cols': 0.75}).to_dict(),
                     DataCheckWarning(message="Row '1' is 50.0% or more null",
                                      data_check_name=HighlyNullRowsDataCheck.name,
                                      message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                      details={'row': 1, 'pct_null_cols': 0.5}).to_dict()],
        'errors': [],
        'actions': [DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 0}).to_dict(),
                    DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 1}).to_dict()]}

    #  test Woodwork
    ww_input = ww.DataTable(pd.DataFrame([[None, None, None, 20], [None, None, "text", "text_1"]]))
    assert highly_null_rows_check.validate(ww_input) == expected

    #  test 2D list
    assert highly_null_rows_check.validate([[None, None, None, 20], [None, None, "text", "text_1"]]) == expected

    # test np.array
    assert highly_null_rows_check.validate(np.array([[None, None, None, 20], [None, None, "text", "text_1"]])) == expected
