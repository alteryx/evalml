import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import TextFeaturization


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


def test_transform_without_fit(text_df):
    X = text_df
    tf = TextFeaturization(text_columns=['col_1', 'col_2'])

    with pytest.raises(RuntimeError, match='You must fit'):
        tf.transform(X)


def test_featurization_only_text(text_df):
    X = text_df
    tf = TextFeaturization(text_columns=['col_1', 'col_2'])

    tf.fit(X)
    expected_features = set(['DIVERSITY_SCORE(col_1)',
                             'DIVERSITY_SCORE(col_2)',
                             'LSA(col_1)',
                             'LSA(col_2)',
                             'MEAN_CHARACTERS_PER_WORD(col_1)',
                             'MEAN_CHARACTERS_PER_WORD(col_2)',
                             'PART_OF_SPEECH_COUNT(col_1)',
                             'PART_OF_SPEECH_COUNT(col_2)',
                             'POLARITY_SCORE(col_1)',
                             'POLARITY_SCORE(col_2)'])
    features = set([feat.get_name() for feat in tf.features])
    assert expected_features == features
    X_t = tf.transform(X)
    assert len(X_t.columns) == 40
    assert X_t.dtypes.all() == np.float64


def test_featurization_with_nontext(text_df):
    X = text_df
    X['col_3'] = [73.7, 67.213, 92]
    tf = TextFeaturization(text_columns=['col_1', 'col_2'])

    tf.fit(X)
    expected_features = set(['DIVERSITY_SCORE(col_1)',
                             'DIVERSITY_SCORE(col_2)',
                             'LSA(col_1)',
                             'LSA(col_2)',
                             'MEAN_CHARACTERS_PER_WORD(col_1)',
                             'MEAN_CHARACTERS_PER_WORD(col_2)',
                             'PART_OF_SPEECH_COUNT(col_1)',
                             'PART_OF_SPEECH_COUNT(col_2)',
                             'POLARITY_SCORE(col_1)',
                             'POLARITY_SCORE(col_2)'])
    features = set([feat.get_name() for feat in tf.features])
    assert expected_features == features
    X_t = tf.transform(X)
    assert len(X_t.columns) == 41
    assert X_t.dtypes.all() == np.float64


def test_featurization_no_text():
    X = pd.DataFrame({'col_1': [1, 2, 3], 'col_2': [4, 5, 6]})
    warn_msg = "No text columns were given to TextFeaturization, component will have no effect"
    with pytest.warns(RuntimeWarning, match=warn_msg):
        tf = TextFeaturization()

    tf.fit(X)
    assert len(tf.features) == 0
    X_t = tf.transform(X)
    assert len(X_t.columns) == 2


def test_some_missing_col_names(text_df):
    X = text_df
    tf = TextFeaturization(text_columns=['col_1', 'col_2', 'col_3'])

    with pytest.warns(RuntimeWarning, match="not found in the given DataFrame"):
        tf.fit(X)

    expected_features = set(['DIVERSITY_SCORE(col_1)',
                             'DIVERSITY_SCORE(col_2)',
                             'LSA(col_1)',
                             'LSA(col_2)',
                             'MEAN_CHARACTERS_PER_WORD(col_1)',
                             'MEAN_CHARACTERS_PER_WORD(col_2)',
                             'PART_OF_SPEECH_COUNT(col_1)',
                             'PART_OF_SPEECH_COUNT(col_2)',
                             'POLARITY_SCORE(col_1)',
                             'POLARITY_SCORE(col_2)'])
    features = set([feat.get_name() for feat in tf.features])
    assert expected_features == features
    X_t = tf.transform(X)
    assert len(X_t.columns) == 40
    assert X_t.dtypes.all() == np.float64


def test_all_missing_col_names(text_df):
    X = text_df
    tf = TextFeaturization(text_columns=['col_3', 'col_4'])

    error_msg = "None of the provided text column names match the columns in the given DataFrame"
    with pytest.raises(RuntimeError, match=error_msg):
        tf.fit(X)

    with pytest.raises(RuntimeError, match="You must fit"):
        tf.transform(X)


def test_invalid_text_column():
    X = pd.DataFrame({'col_1': []})
    tf = TextFeaturization(text_columns=['col_1'])
    with pytest.raises(ValueError, match="not a text column"):
        tf.fit(X)

    X = pd.DataFrame(
        {'col_1': [
            'I\'m singing in the rain!$%^ do do do do do da do',
            'just singing in the rain.................. \n',
            325,
            'I\'m happy again!!! lalalalalalalalalalala']})
    tf = TextFeaturization(text_columns=['col_1'])
    with pytest.raises(ValueError, match="not a text column"):
        tf.fit(X)


def test_index_col_names():
    X = np.array([['I\'m singing in the rain!$%^ do do do do do da do', 'do you hear the people sing?////////////////////////////////////'],
                  ['just singing in the rain.................. \n', 'singing the songs of angry men\n'],
                  ['\t\n\n\n\nWhat a glorious feelinggggggggggg, I\'m happy again!!! lalalalalalalalalalala', '\tIt is the music of a people who will NOT be slaves again!!!!!!!!!!!']])
    tf = TextFeaturization(text_columns=[0, 1])

    tf.fit(X)
    expected_features = set(['DIVERSITY_SCORE(0)',
                             'DIVERSITY_SCORE(1)',
                             'LSA(0)',
                             'LSA(1)',
                             'MEAN_CHARACTERS_PER_WORD(0)',
                             'MEAN_CHARACTERS_PER_WORD(1)',
                             'PART_OF_SPEECH_COUNT(0)',
                             'PART_OF_SPEECH_COUNT(1)',
                             'POLARITY_SCORE(0)',
                             'POLARITY_SCORE(1)'])
    features = set([feat.get_name() for feat in tf.features])
    assert expected_features == features
    X_t = tf.transform(X)
    assert len(X_t.columns) == 40
    assert X_t.dtypes.all() == np.float64
