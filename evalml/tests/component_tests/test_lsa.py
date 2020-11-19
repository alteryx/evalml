import logging

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import LSA


@pytest.fixture()
def text_df():
    df = pd.DataFrame(
        {'col_1': ['I\'m singing in the rain! Just singing in the rain, what a glorious feeling, I\'m happy again!',
                   'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
                   'I\'m gonna be the main event, like no king was before! I\'m brushing up on looking down, I\'m working on my ROAR!'],
         'col_2': ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
                   'I dreamed a dream in days gone by, when hope was high and life worth living',
                   'Red, the blood of angry men - black, the dark of ages past']
         })
    yield df


def test_lsa_only_text(text_df):
    X = text_df
    lsa = LSA(text_columns=['col_1', 'col_2'])
    lsa.fit(X)

    expected_col_names = set(['LSA(col_1)[0]',
                              'LSA(col_1)[1]',
                              'LSA(col_2)[0]',
                              'LSA(col_2)[1]'])
    X_t = lsa.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 4
    assert X_t.dtypes.all() == np.float64


def test_lsa_with_nontext(text_df):
    X = text_df
    X['col_3'] = [73.7, 67.213, 92]
    lsa = LSA(text_columns=['col_1', 'col_2'])

    lsa.fit(X)
    expected_col_names = set(['LSA(col_1)[0]',
                              'LSA(col_1)[1]',
                              'LSA(col_2)[0]',
                              'LSA(col_2)[1]',
                              'col_3'])
    X_t = lsa.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 5
    assert X_t.dtypes.all() == np.float64


def test_lsa_no_text():
    X = pd.DataFrame({'col_1': [1, 2, 3], 'col_2': [4, 5, 6]})
    lsa = LSA()
    lsa.fit(X)
    X_t = lsa.transform(X)
    assert len(X_t.columns) == 2


def test_some_missing_col_names(text_df, caplog):
    X = text_df
    lsa = LSA(text_columns=['col_1', 'col_2', 'col_3'])

    with caplog.at_level(logging.WARNING):
        lsa.fit(X)
    assert "Columns ['col_3'] were not found in the given DataFrame, ignoring" in caplog.messages

    expected_col_names = set(['LSA(col_1)[0]',
                              'LSA(col_1)[1]',
                              'LSA(col_2)[0]',
                              'LSA(col_2)[1]'])
    X_t = lsa.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 4
    assert X_t.dtypes.all() == np.float64


def test_all_missing_col_names(text_df):
    X = text_df
    lsa = LSA(text_columns=['col_3', 'col_4'])

    error_msg = "None of the provided text column names match the columns in the given DataFrame"
    with pytest.raises(AttributeError, match=error_msg):
        lsa.fit(X)


def test_empty_text_column():
    X = pd.DataFrame({'col_1': []})
    lsa = LSA(text_columns=['col_1'])
    with pytest.raises(ValueError, match="empty vocabulary"):
        lsa.fit(X)


def test_invalid_text_column():
    X = pd.DataFrame({'col_1': []})
    lsa = LSA(text_columns=['col_1'])
    with pytest.raises(ValueError, match="empty vocabulary; perhaps the documents only contain stop words"):
        lsa.fit(X)

    # we assume this sort of data would fail to validate as text data up the stack
    # but just in case, make sure our component will convert non-str values to str
    X = pd.DataFrame(
        {'col_1': [
            'I\'m singing in the rain!$%^ do do do do do da do',
            'just singing in the rain.................. \n',
            325,
            np.nan,
            None,
            'I\'m happy again!!! lalalalalalalalalalala']})
    lsa = LSA(text_columns=['col_1'])
    lsa.fit(X)


def test_index_col_names():
    X = np.array([['I\'m singing in the rain!$%^ do do do do do da do', 'do you hear the people sing?////////////////////////////////////'],
                  ['just singing in the rain.................. \n', 'singing the songs of angry men\n'],
                  ['\t\n\n\n\nWhat a glorious feelinggggggggggg, I\'m happy again!!! lalalalalalalalalalala', '\tIt is the music of a people who will NOT be slaves again!!!!!!!!!!!']])
    lsa = LSA(text_columns=[0, 1])

    lsa.fit(X)
    expected_col_names = set(['LSA(0)[0]',
                              'LSA(0)[1]',
                              'LSA(1)[0]',
                              'LSA(1)[1]'])
    X_t = lsa.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 4
    assert X_t.dtypes.all() == np.float64


def test_int_col_names():
    X = pd.DataFrame(
        {4.75: ['I\'m singing in the rain! Just singing in the rain, what a glorious feeling, I\'m happy again!',
                'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
                'I\'m gonna be the main event, like no king was before! I\'m brushing up on looking down, I\'m working on my ROAR!'],
         -1: ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
              'I dreamed a dream in days gone by, when hope was high and life worth living',
              'Red, the blood of angry men - black, the dark of ages past']
         })
    lsa = LSA(text_columns=[4.75, -1])
    lsa.fit(X)
    expected_col_names = set(['LSA(4.75)[0]',
                              'LSA(4.75)[1]',
                              'LSA(-1)[0]',
                              'LSA(-1)[1]'])
    X_t = lsa.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 4
    assert X_t.dtypes.all() == np.float64


def test_lsa_output():
    X = pd.DataFrame(
        {'lsa': ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
                 'I dreamed a dream in days gone by, when hope was high and life worth living',
                 'Red, the blood of angry men - black, the dark of ages past']})
    lsa = LSA(text_columns=['lsa'])
    lsa.fit(X)

    expected_features = [[0.832, 0.],
                         [0., 1.],
                         [0.832, 0.]]
    X_t = lsa.transform(X)
    cols = [col for col in X_t.columns if 'LSA' in col]
    features = X_t[cols]
    np.testing.assert_almost_equal(features, expected_features, decimal=3)
