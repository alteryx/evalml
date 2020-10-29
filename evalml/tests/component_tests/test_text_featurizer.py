import logging

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import TextFeaturizer


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


def test_featurizer_only_text(text_df):
    X = text_df
    tf = TextFeaturizer(text_columns=['col_1', 'col_2'])
    tf.fit(X)

    expected_col_names = set(['DIVERSITY_SCORE(col_1)',
                              'DIVERSITY_SCORE(col_2)',
                              'LSA(col_1)[0]',
                              'LSA(col_1)[1]',
                              'LSA(col_2)[0]',
                              'LSA(col_2)[1]',
                              'MEAN_CHARACTERS_PER_WORD(col_1)',
                              'MEAN_CHARACTERS_PER_WORD(col_2)',
                              'POLARITY_SCORE(col_1)',
                              'POLARITY_SCORE(col_2)'])
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 10
    assert X_t.dtypes.all() == np.float64


def test_featurizer_with_nontext(text_df):
    X = text_df
    X['col_3'] = [73.7, 67.213, 92]
    tf = TextFeaturizer(text_columns=['col_1', 'col_2'])

    tf.fit(X)
    expected_col_names = set(['DIVERSITY_SCORE(col_1)',
                              'DIVERSITY_SCORE(col_2)',
                              'LSA(col_1)[0]',
                              'LSA(col_1)[1]',
                              'LSA(col_2)[0]',
                              'LSA(col_2)[1]',
                              'MEAN_CHARACTERS_PER_WORD(col_1)',
                              'MEAN_CHARACTERS_PER_WORD(col_2)',
                              'POLARITY_SCORE(col_1)',
                              'POLARITY_SCORE(col_2)',
                              'col_3'])
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 11
    assert X_t.dtypes.all() == np.float64


def test_featurizer_no_text():
    X = pd.DataFrame({'col_1': [1, 2, 3], 'col_2': [4, 5, 6]})
    tf = TextFeaturizer()
    tf.fit(X)
    X_t = tf.transform(X)
    assert len(X_t.columns) == 2


def test_some_missing_col_names(text_df, caplog):
    X = text_df
    tf = TextFeaturizer(text_columns=['col_1', 'col_2', 'col_3'])

    with caplog.at_level(logging.WARNING):
        tf.fit(X)
    assert "Columns ['col_3'] were not found in the given DataFrame, ignoring" in caplog.messages

    expected_col_names = set(['DIVERSITY_SCORE(col_1)',
                              'DIVERSITY_SCORE(col_2)',
                              'LSA(col_1)[0]',
                              'LSA(col_1)[1]',
                              'LSA(col_2)[0]',
                              'LSA(col_2)[1]',
                              'MEAN_CHARACTERS_PER_WORD(col_1)',
                              'MEAN_CHARACTERS_PER_WORD(col_2)',
                              'POLARITY_SCORE(col_1)',
                              'POLARITY_SCORE(col_2)'])
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 10
    assert X_t.dtypes.all() == np.float64


def test_all_missing_col_names(text_df):
    X = text_df
    tf = TextFeaturizer(text_columns=['col_3', 'col_4'])

    error_msg = "None of the provided text column names match the columns in the given DataFrame"
    with pytest.raises(AttributeError, match=error_msg):
        tf.fit(X)


def test_invalid_text_column():
    X = pd.DataFrame({'col_1': []})
    tf = TextFeaturizer(text_columns=['col_1'])
    with pytest.raises(ValueError, match="empty vocabulary; perhaps the documents only contain stop words"):
        tf.fit(X)

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
    tf = TextFeaturizer(text_columns=['col_1'])
    tf.fit(X)


def test_no_null_output():
    X = pd.DataFrame(
        {'col_1': ['I\'m singing in the rain! Just singing in the rain, what a glorious feeling, I\'m happy again!',
                   'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
                   'I\'m gonna be the main event, like no king was before! I\'m brushing up on looking down, I\'m working on my ROAR!'],
         'col_2': ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
                   'I dreamed a dream in days gone by, when hope was high and life worth living Red, the blood of angry men - black, the dark of ages past',
                   ':)']
         })
    tf = TextFeaturizer(text_columns=['col_1', 'col_2'])
    tf.fit(X)
    X_t = tf.transform(X)
    assert not X_t.isnull().any().any()


def test_index_col_names():
    X = np.array([['I\'m singing in the rain!$%^ do do do do do da do', 'do you hear the people sing?////////////////////////////////////'],
                  ['just singing in the rain.................. \n', 'singing the songs of angry men\n'],
                  ['\t\n\n\n\nWhat a glorious feelinggggggggggg, I\'m happy again!!! lalalalalalalalalalala', '\tIt is the music of a people who will NOT be slaves again!!!!!!!!!!!']])
    tf = TextFeaturizer(text_columns=[0, 1])

    tf.fit(X)
    expected_col_names = set(['DIVERSITY_SCORE(0)',
                              'DIVERSITY_SCORE(1)',
                              'LSA(0)[0]',
                              'LSA(0)[1]',
                              'LSA(1)[0]',
                              'LSA(1)[1]',
                              'MEAN_CHARACTERS_PER_WORD(0)',
                              'MEAN_CHARACTERS_PER_WORD(1)',
                              'POLARITY_SCORE(0)',
                              'POLARITY_SCORE(1)'])
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 10
    assert X_t.dtypes.all() == np.float64


def test_int_col_names():
    X = pd.DataFrame(
        {475: ['I\'m singing in the rain! Just singing in the rain, what a glorious feeling, I\'m happy again!',
               'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
               'I\'m gonna be the main event, like no king was before! I\'m brushing up on looking down, I\'m working on my ROAR!'],
         -1: ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
              'I dreamed a dream in days gone by, when hope was high and life worth living',
              'Red, the blood of angry men - black, the dark of ages past']
         })
    tf = TextFeaturizer(text_columns=[475, -1])
    tf.fit(X)
    expected_col_names = set(['DIVERSITY_SCORE(475)',
                              'DIVERSITY_SCORE(-1)',
                              'LSA(475)[0]',
                              'LSA(475)[1]',
                              'LSA(-1)[0]',
                              'LSA(-1)[1]',
                              'MEAN_CHARACTERS_PER_WORD(475)',
                              'MEAN_CHARACTERS_PER_WORD(-1)',
                              'POLARITY_SCORE(475)',
                              'POLARITY_SCORE(-1)'])
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 10
    assert X_t.dtypes.all() == np.float64


def test_output_null():
    X = pd.DataFrame(
        {'col_1': ['I\'m singing in the rain! Just singing in the rain, what a glorious feeling, I\'m happy again!',
                   'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
                   'I\'m gonna be the main event, like no king was before! I\'m brushing up on looking down, I\'m working on my ROAR!'],
         'col_2': ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
                   'I dreamed a dream in days gone by, when hope was high and life worth living Red, the blood of angry men - black, the dark of ages past',
                   ':)']
         })
    tf = TextFeaturizer(text_columns=['col_1', 'col_2'])
    tf.fit(X)
    X_t = tf.transform(X)
    assert not X_t.isnull().any().any()


def test_diversity_primitive_output():
    X = pd.DataFrame(
        {'diverse': ['This is a very diverse string which does not contain any repeated words at all',
                     'Here here each each word word is is repeated repeated exactly exactly twice twice',
                     'A sentence sentence with just a little overlap here and there there there']})
    tf = TextFeaturizer(text_columns=['diverse'])
    tf.fit(X)

    expected_features = [1.0, 0.5, 0.75]
    X_t = tf.transform(X)
    features = X_t['DIVERSITY_SCORE(diverse)']
    np.testing.assert_almost_equal(features, expected_features)


def test_lsa_primitive_output():
    X = pd.DataFrame(
        {'lsa': ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
                 'I dreamed a dream in days gone by, when hope was high and life worth living',
                 'Red, the blood of angry men - black, the dark of ages past']})
    tf = TextFeaturizer(text_columns=['lsa'])
    tf.fit(X)

    expected_features = [[0.832, 0.],
                         [0., 1.],
                         [0.832, 0.]]
    X_t = tf.transform(X)
    cols = [col for col in X_t.columns if 'LSA' in col]
    features = X_t[cols]
    np.testing.assert_almost_equal(features, expected_features, decimal=3)


def test_mean_characters_primitive_output():
    X = pd.DataFrame(
        {'mean_characters': ['I\'m singing in the rain! Just singing in the rain, what a glorious feeling, I\'m happy again!',
                             'In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.',
                             'I\'m gonna be the main event, like no king was before! I\'m brushing up on looking down, I\'m working on my ROAR!']})
    tf = TextFeaturizer(text_columns=['mean_characters'])
    tf.fit(X)

    expected_features = [4.11764705882352, 3.45, 3.72727272727]
    X_t = tf.transform(X)
    features = X_t['MEAN_CHARACTERS_PER_WORD(mean_characters)']
    np.testing.assert_almost_equal(features, expected_features)


def test_polarity_primitive_output():
    X = pd.DataFrame(
        {'polarity': ['This is neutral.',
                      'Everything is bad. Nothing is happy, he hates milk and can\'t stand gross foods, we are being very negative.',
                      'Everything is awesome! Everything is cool when you\'re part of a team! He loves milk and cookies!']})
    tf = TextFeaturizer(text_columns=['polarity'])
    tf.fit(X)

    expected_features = [0.0, -0.214, 0.602]
    X_t = tf.transform(X)
    features = X_t['POLARITY_SCORE(polarity)']
    np.testing.assert_almost_equal(features, expected_features)
