import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import TextFeaturization


@pytest.fixture()
def text_df():
    df = pd.DataFrame(
        {'col_1': ['I\'m singing in the rain!$%^ do do do do do da do', 'just singing in the rain.................. \n', '\t\n\n\n\nWhat a glorious feelinggggggggggg, I\'m happy again!!! lalalalalalalalalalala'],
         'col_2': ['do you hear the people sing?////////////////////////////////////', 'singing the songs of angry men\n', '\tIt is the music of a people who will NOT be slaves again!!!!!!!!!!!']})
    yield df


def test_invalid_col_name():
    with pytest.raises(ValueError, match="Column names must be of object type"):
        TextFeaturization(text_columns=['col_1', 2])


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
