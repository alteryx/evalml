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


def test_transform_without_fit(text_df):
    X = text_df
    tf = TextFeaturizer(text_columns=['col_1', 'col_2'])

    with pytest.raises(RuntimeError, match='You must fit'):
        tf.transform(X)


def test_featurization_only_text(text_df):
    X = text_df
    tf = TextFeaturizer(text_columns=['col_1', 'col_2'])

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
    tf = TextFeaturizer(text_columns=['col_1', 'col_2'])

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
    warn_msg = "No text columns were given to TextFeaturizer, component will have no effect"
    with pytest.warns(RuntimeWarning, match=warn_msg):
        tf = TextFeaturizer()

    tf.fit(X)
    assert len(tf.features) == 0
    X_t = tf.transform(X)
    assert len(X_t.columns) == 2


def test_some_missing_col_names(text_df):
    X = text_df
    tf = TextFeaturizer(text_columns=['col_1', 'col_2', 'col_3'])

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
    tf = TextFeaturizer(text_columns=['col_3', 'col_4'])

    error_msg = "None of the provided text column names match the columns in the given DataFrame"
    with pytest.raises(RuntimeError, match=error_msg):
        tf.fit(X)

    with pytest.raises(RuntimeError, match="You must fit"):
        tf.transform(X)


def test_invalid_text_column():
    X = pd.DataFrame({'col_1': []})
    tf = TextFeaturizer(text_columns=['col_1'])
    with pytest.raises(ValueError, match="not a text column"):
        tf.fit(X)

    X = pd.DataFrame(
        {'col_1': [
            'I\'m singing in the rain!$%^ do do do do do da do',
            'just singing in the rain.................. \n',
            325,
            'I\'m happy again!!! lalalalalalalalalalala']})
    tf = TextFeaturizer(text_columns=['col_1'])
    with pytest.raises(ValueError, match="not a text column"):
        tf.fit(X)


def test_index_col_names():
    X = np.array([['I\'m singing in the rain!$%^ do do do do do da do', 'do you hear the people sing?////////////////////////////////////'],
                  ['just singing in the rain.................. \n', 'singing the songs of angry men\n'],
                  ['\t\n\n\n\nWhat a glorious feelinggggggggggg, I\'m happy again!!! lalalalalalalalalalala', '\tIt is the music of a people who will NOT be slaves again!!!!!!!!!!!']])
    tf = TextFeaturizer(text_columns=[0, 1])

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

    expected_features = [[0.0200961, 0.002976],
                         [0.0223392, 0.0058817],
                         [0.0186072, -0.0006121]]
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


def test_part_of_speech_primitive_output():
    X = pd.DataFrame(
        {'part_of_speech': ['do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!',
                            'I dreamed a dream in days gone by, when hope was high and life worth living',
                            'Red, the blood of angry men - black, the dark of ages past']})
    tf = TextFeaturizer(text_columns=['part_of_speech'])
    tf.fit(X)

    expected_features = [[0, 0, 0, 0, 0, 2, 0, 0, 6, 0, 0, 0, 0, 2, 0],
                         [0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 0, 0, 3, 0],
                         [0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0]]
    X_t = tf.transform(X)
    cols = [col for col in X_t.columns if 'PART_OF_SPEECH' in col]
    features = X_t[cols]
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
