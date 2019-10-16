import pandas as pd

from evalml.preprocessing import detect_id_columns


def test_detect_id_columns():
    X_dict = {'col_1_id': [0, 1, 2, 3],
              'col_2': [2, 3, 4, 5],
              'col_3_id': [1, 1, 2, 3],
              'Id': [3, 1, 2, 0],
              'col_5': [0, 0, 1, 2]}
    X = pd.DataFrame.from_dict(X_dict)

    expected = {'Id': 1.0, 'col_1_id': 1.0, 'col_2': 0.95, 'col_3_id': 0.95}
    result = detect_id_columns(X, 0.95)
    assert expected == result

    expected = {'Id': 1.0, 'col_1_id': 1.0}
    result = detect_id_columns(X)
    assert expected == result
