import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    MulticollinearityDataCheck
)

multi_data_check_name = MulticollinearityDataCheck.name


def test_multicollinearity_data_check_init():
    multi_check = MulticollinearityDataCheck()
    assert multi_check.threshold == 1.0

    multi_check = MulticollinearityDataCheck(threshold=0.0)
    assert multi_check.threshold == 0

    multi_check = MulticollinearityDataCheck(threshold=0.5)
    assert multi_check.threshold == 0.5

    multi_check = MulticollinearityDataCheck(threshold=1.0)
    assert multi_check.threshold == 1.0

    with pytest.raises(ValueError, match="threshold must be a float between 0 and 1, inclusive."):
        MulticollinearityDataCheck(threshold=-0.1)
    with pytest.raises(ValueError, match="threshold must be a float between 0 and 1, inclusive."):
        MulticollinearityDataCheck(threshold=1.1)


def test_multicollinearity_returns_warning():
    col = pd.Series([1, 0, 2, 3, 4])
    X = pd.DataFrame({'col_1': col,
                      'col_2': col * 3,
                      'col_3': ~col,
                      'col_4': col / 2,
                      'col_5': col + 1,
                      'not_collinear': [0, 1, 0, 0, 0]})

    multi_check = MulticollinearityDataCheck(threshold=0.95)
    assert multi_check.validate(X) == {
        "warnings": [DataCheckWarning(message="Columns are likely to be correlated: [('col_1', 'col_2'), ('col_1', 'col_3'), ('col_1', 'col_4'), ('col_1', 'col_5'), ('col_2', 'col_3'), ('col_2', 'col_4'), ('col_2', 'col_5'), ('col_3', 'col_4'), ('col_3', 'col_5'), ('col_4', 'col_5')]",
                                      data_check_name=multi_data_check_name,
                                      message_code=DataCheckMessageCode.IS_MULTICOLLINEAR,
                                      details={'columns': [('col_1', 'col_2'), ('col_1', 'col_3'), ('col_1', 'col_4'), ('col_1', 'col_5'),
                                                           ('col_2', 'col_3'), ('col_2', 'col_4'), ('col_2', 'col_5'),
                                                           ('col_3', 'col_4'), ('col_3', 'col_5'), ('col_4', 'col_5')]}).to_dict()],
        "errors": []
    }


def test_id_columns_strings():
    X_dict = {'col_1_id': ["a", "b", "c", "d"],
              'col_2': ["w", "x", "y", "z"],
              'col_3_id': ["a", "a", "b", "d"],
              'Id': ["z", "y", "x", "a"],
              'col_5': ["0", "0", "1", "2"],
              'col_6': [0.1, 0.2, 0.3, 0.4]
              }
    X = pd.DataFrame.from_dict(X_dict)
    multi_check = MulticollinearityDataCheck(threshold=0.95)
    assert multi_check.validate(X) == {
        "warnings": [],
        "errors": []
    }

    multi_check = MulticollinearityDataCheck(threshold=1.0)
    assert multi_check.validate(X) == {
        "warnings": [],
        "errors": []
    }


def test_multicollinearity_data_check_input_formats():
    multi_check = MulticollinearityDataCheck(threshold=0.8)

    # test empty pd.DataFrame
    assert multi_check.validate(pd.DataFrame()) == {"warnings": [], "errors": []}

    #  test Woodwork
    ww_input = ww.DataTable(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]))
    assert multi_check.validate(ww_input) == {
        "warnings": [],
        "errors": []
    }

    # test np.array
    # may need next release to work
    # assert multi_check.validate(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])) == {
    #     "warnings": [],
    #     "errors": []
    # }
