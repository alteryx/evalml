from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_series_equal
from woodwork.logical_types import Boolean, Categorical, Double, Integer

from evalml.pipelines.components import NaturalLanguageFeaturizer
from evalml.utils import infer_feature_types


def test_featurizer_only_text(text_df):
    X = text_df
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)

    expected_col_names = set(
        [
            "DIVERSITY_SCORE(col_1)",
            "DIVERSITY_SCORE(col_2)",
            "LSA(col_1)[0]",
            "LSA(col_1)[1]",
            "LSA(col_2)[0]",
            "LSA(col_2)[1]",
            "MEAN_CHARACTERS_PER_WORD(col_1)",
            "MEAN_CHARACTERS_PER_WORD(col_2)",
            "NUM_CHARACTERS(col_1)",
            "NUM_CHARACTERS(col_2)",
            "NUM_WORDS(col_1)",
            "NUM_WORDS(col_2)",
            "POLARITY_SCORE(col_1)",
            "POLARITY_SCORE(col_2)",
        ],
    )
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 14
    assert set([type(v) for v in X_t.ww.logical_types.values()]) == {
        ww.logical_types.Double,
    }


def test_featurizer_with_nontext(text_df):
    X = text_df
    X["col_3"] = [73.7, 67.213, 92]
    X.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()

    tf.fit(X)
    expected_col_names = set(
        [
            "DIVERSITY_SCORE(col_1)",
            "DIVERSITY_SCORE(col_2)",
            "LSA(col_1)[0]",
            "LSA(col_1)[1]",
            "LSA(col_2)[0]",
            "LSA(col_2)[1]",
            "MEAN_CHARACTERS_PER_WORD(col_1)",
            "MEAN_CHARACTERS_PER_WORD(col_2)",
            "NUM_CHARACTERS(col_1)",
            "NUM_CHARACTERS(col_2)",
            "NUM_WORDS(col_1)",
            "NUM_WORDS(col_2)",
            "POLARITY_SCORE(col_1)",
            "POLARITY_SCORE(col_2)",
            "col_3",
        ],
    )
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 15
    assert set([type(v) for v in X_t.ww.logical_types.values()]) == {
        ww.logical_types.Double,
    }


def test_featurizer_no_text():
    X = pd.DataFrame({"col_1": [1, 2, 3], "col_2": [4, 5, 6]})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)
    X_t = tf.transform(X)
    assert len(X_t.columns) == 2


def test_some_missing_col_names(text_df, caplog):
    X = text_df
    tf = NaturalLanguageFeaturizer(text_columns=["col_1", "col_2", "col_3"])
    expected_col_names = set(
        [
            "DIVERSITY_SCORE(col_1)",
            "DIVERSITY_SCORE(col_2)",
            "LSA(col_1)[0]",
            "LSA(col_1)[1]",
            "LSA(col_2)[0]",
            "LSA(col_2)[1]",
            "MEAN_CHARACTERS_PER_WORD(col_1)",
            "MEAN_CHARACTERS_PER_WORD(col_2)",
            "NUM_CHARACTERS(col_1)",
            "NUM_CHARACTERS(col_2)",
            "NUM_WORDS(col_1)",
            "NUM_WORDS(col_2)",
            "POLARITY_SCORE(col_1)",
            "POLARITY_SCORE(col_2)",
        ],
    )
    tf.fit(X)
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 14
    assert set([type(v) for v in X_t.ww.logical_types.values()]) == {
        ww.logical_types.Double,
    }


def test_empty_text_column():
    X = pd.DataFrame({"col_1": []})
    X = infer_feature_types(X, {"col_1": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    with pytest.raises(
        ValueError,
        match="empty vocabulary; perhaps the documents only contain stop words",
    ):
        tf.fit(X)


def test_invalid_text_column():
    # we assume this sort of data would fail to validate as text data up the stack
    # but just in case, make sure our component will convert non-str values to str
    X = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain!$%^ do do do do do da do",
                "just singing in the rain.................. \n",
                325,
                np.nan,
                None,
                "I'm happy again!!! lalalalalalalalalalala",
            ],
        },
    )
    X = infer_feature_types(X, {"col_1": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)


def test_no_null_output():
    X = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
            ],
            "col_2": [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living Red, the blood of angry men - black, the dark of ages past",
                ":)",
            ],
        },
    )
    X.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)
    X_t = tf.transform(X)
    assert not X_t.isnull().any().any()


def test_index_col_names():
    X = np.array(
        [
            [
                "I'm singing in the rain!$%^ do do do do do da do",
                "do you hear the people sing?////////////////////////////////////",
            ],
            [
                "just singing in the rain.................. \n",
                "singing the songs of angry men\n",
            ],
            [
                "\t\n\n\n\nWhat a glorious feelinggggggggggg, I'm happy again!!! lalalalalalalalalalala",
                "\tIt is the music of a people who will NOT be slaves again!!!!!!!!!!!",
            ],
        ],
    )
    X = pd.DataFrame(X)
    X.ww.init(logical_types={0: "NaturalLanguage", 1: "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()

    tf.fit(X)
    expected_col_names = set(
        [
            "DIVERSITY_SCORE(0)",
            "DIVERSITY_SCORE(1)",
            "LSA(0)[0]",
            "LSA(0)[1]",
            "LSA(1)[0]",
            "LSA(1)[1]",
            "MEAN_CHARACTERS_PER_WORD(0)",
            "MEAN_CHARACTERS_PER_WORD(1)",
            "NUM_CHARACTERS(0)",
            "NUM_CHARACTERS(1)",
            "NUM_WORDS(0)",
            "NUM_WORDS(1)",
            "POLARITY_SCORE(0)",
            "POLARITY_SCORE(1)",
        ],
    )
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 14
    assert set([type(v) for v in X_t.ww.logical_types.values()]) == {
        ww.logical_types.Double,
    }


def test_float_col_names():
    X = pd.DataFrame(
        {
            4.75: [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
            ],
            -1: [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living",
                "Red, the blood of angry men - black, the dark of ages past",
            ],
        },
    )
    X.ww.init(logical_types={4.75: "NaturalLanguage", -1: "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)
    expected_col_names = set(
        [
            "DIVERSITY_SCORE(4.75)",
            "DIVERSITY_SCORE(-1.0)",
            "LSA(4.75)[0]",
            "LSA(4.75)[1]",
            "LSA(-1.0)[0]",
            "LSA(-1.0)[1]",
            "MEAN_CHARACTERS_PER_WORD(4.75)",
            "MEAN_CHARACTERS_PER_WORD(-1.0)",
            "NUM_CHARACTERS(4.75)",
            "NUM_CHARACTERS(-1.0)",
            "NUM_WORDS(4.75)",
            "NUM_WORDS(-1.0)",
            "POLARITY_SCORE(4.75)",
            "POLARITY_SCORE(-1.0)",
        ],
    )
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    assert len(X_t.columns) == 14
    assert set([type(v) for v in X_t.ww.logical_types.values()]) == {
        ww.logical_types.Double,
    }


def test_output_null():
    X = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
            ],
            "col_2": [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living Red, the blood of angry men - black, the dark of ages past",
                ":)",
            ],
        },
    )
    X.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)
    X_t = tf.transform(X)
    assert not X_t.isnull().any().any()


def test_diversity_primitive_output():
    X = pd.DataFrame(
        {
            "diverse": [
                "This is a very diverse string which does not contain any repeated words at all",
                "Here here each each word word is is repeated repeated exactly exactly twice twice",
                "A sentence sentence with just a little overlap here and there there there",
            ],
        },
    )
    X.ww.init(logical_types={"diverse": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)

    expected_features = pd.Series([1.0, 0.5, 0.75], name="DIVERSITY_SCORE(diverse)")
    X_t = tf.transform(X)
    features = X_t["DIVERSITY_SCORE(diverse)"]
    assert_series_equal(expected_features, features)


def test_lsa_primitive_output():
    X = pd.DataFrame(
        {
            "lsa": [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living",
                "Red, the blood of angry men - black, the dark of ages past",
            ],
        },
    )
    X.ww.init(logical_types={"lsa": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)

    expected_features = pd.DataFrame(
        [[0.832, 0.0], [0.0, 1.0], [0.832, 0.0]],
        columns=["LSA(lsa)[0]", "LSA(lsa)[1]"],
    )
    X_t = tf.transform(X)
    cols = [col for col in X_t.columns if "LSA" in col]
    features = X_t[cols]
    assert_frame_equal(expected_features, features, atol=1e-3)


def test_featurizer_custom_types(text_df):
    # force one of the two provided columns to be a user-specified type.
    # if the output contains text features for col_2, then the natural language featurizer didn't pass the right
    # ww types to LSA, because LSA still thought col_2 was natural language even though the user said otherwise.
    X = infer_feature_types(
        pd.DataFrame(text_df),
        {"col_1": "NaturalLanguage", "col_2": "categorical"},
    )
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)

    expected_col_names = set(
        [
            "col_2",
            "DIVERSITY_SCORE(col_1)",
            "LSA(col_1)[0]",
            "LSA(col_1)[1]",
            "MEAN_CHARACTERS_PER_WORD(col_1)",
            "NUM_CHARACTERS(col_1)",
            "NUM_WORDS(col_1)",
            "POLARITY_SCORE(col_1)",
        ],
    )
    X_t = tf.transform(X)
    assert set(X_t.columns) == expected_col_names
    expected_logical_types = {
        "col_2": Categorical,
        "DIVERSITY_SCORE(col_1)": Double,
        "MEAN_CHARACTERS_PER_WORD(col_1)": Double,
        "NUM_CHARACTERS(col_1)": Double,
        "NUM_WORDS(col_1)": Double,
        "POLARITY_SCORE(col_1)": Double,
        "LSA(col_1)[0]": Double,
        "LSA(col_1)[1]": Double,
    }
    assert {
        k: type(v) for k, v in X_t.ww.logical_types.items()
    } == expected_logical_types


def test_mean_characters_primitive_output():
    X = pd.DataFrame(
        {
            "mean_characters": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
            ],
        },
    )
    X.ww.init(logical_types={"mean_characters": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)

    expected_features = pd.Series(
        [4.11764705882352, 3.45, 3.72727272727],
        name="MEAN_CHARACTERS_PER_WORD(mean_characters)",
    )
    X_t = tf.transform(X)
    features = X_t["MEAN_CHARACTERS_PER_WORD(mean_characters)"]
    assert_series_equal(expected_features, features)


def test_polarity_primitive_output():
    X = pd.DataFrame(
        {
            "polarity": [
                "This is neutral.",
                "Everything is bad. Nothing is happy, he hates milk and can't stand gross foods, we are being very negative.",
                "Everything is awesome! Everything is cool when you're part of a team! He loves milk and cookies!",
            ],
        },
    )
    X.ww.init(logical_types={"polarity": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)

    expected_features = pd.Series([0.0, -0.214, 0.602], name="POLARITY_SCORE(polarity)")
    X_t = tf.transform(X)
    features = X_t["POLARITY_SCORE(polarity)"]
    assert_series_equal(expected_features, features)


def test_featurizer_with_custom_indices(text_df):
    X = text_df
    X = X.set_index(pd.Series([2, 5, 19]))
    tf = NaturalLanguageFeaturizer(text_columns=["col_1", "col_2"])
    tf.fit(X)
    X_t = tf.transform(X)
    assert not X_t.isnull().any().any()


def test_natural_language_featurizer_does_not_modify_input_data(text_df):
    X = text_df
    expected = X.copy()
    tf = NaturalLanguageFeaturizer(text_columns=["col_1", "col_2"])
    _ = tf.fit_transform(X)
    pd.testing.assert_frame_equal(X, expected)


@pytest.mark.parametrize(
    "X_df",
    [
        pd.DataFrame(pd.Series([1, 2, 10], dtype="Int64")),
        pd.DataFrame(pd.Series([1.0, 2.0, 10.0], dtype="float")),
        pd.DataFrame(pd.Series(["a", "b", "ab"], dtype="category")),
        pd.DataFrame(pd.Series([True, False, True], dtype="boolean")),
        pd.DataFrame(
            pd.Series(
                ["this will be a natural language column because length", "yay", "hay"],
                dtype="string",
            ),
        ),
    ],
)
def test_natural_language_featurizer_woodwork_custom_overrides_returned_by_components(
    X_df,
):
    X_df = X_df.copy()
    X_df["text col"] = pd.Series(
        ["this will be a natural language column because length", "yay", "hay"],
        dtype="string",
    )
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, Boolean]
    tf = NaturalLanguageFeaturizer()

    for logical_type in override_types:
        try:
            X = X_df.copy()
            X.ww.init(logical_types={0: logical_type, "text col": "NaturalLanguage"})
        except ww.exceptions.TypeConversionError:
            continue

        tf.fit(X)
        transformed = tf.transform(X, y)
        assert isinstance(transformed, pd.DataFrame)
        assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
            0: logical_type,
            "LSA(text col)[0]": Double,
            "LSA(text col)[1]": Double,
            "DIVERSITY_SCORE(text col)": Double,
            "MEAN_CHARACTERS_PER_WORD(text col)": Double,
            "NUM_CHARACTERS(text col)": Double,
            "NUM_WORDS(text col)": Double,
            "POLARITY_SCORE(text col)": Double,
        }


@patch("featuretools.dfs")
def test_natural_language_featurizer_sets_max_depth_1(mock_dfs):
    X = pd.DataFrame(
        {
            "polarity": [
                "This is neutral.",
                "Everything is bad. Nothing is happy, he hates milk and can't stand gross foods, we are being very negative.",
                "Everything is awesome! Everything is cool when you're part of a team! He loves milk and cookies!",
            ],
        },
    )
    X.ww.init(logical_types={"polarity": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)
    _, kwargs = mock_dfs.call_args
    assert kwargs["max_depth"] == 1


@pytest.mark.parametrize("nones", [np.nan, pd.NA, None])
def test_nan_allowed(nones):
    X = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
                "",
                nones,
            ],
            "col_2": [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living Red, the blood of angry men - black, the dark of ages past",
                ":)",
                "None",
                "",
            ],
        },
    )
    X.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)
    X_t = tf.transform(X)
    cols = [
        "LSA(col_1)[0]",
        "LSA(col_1)[1]",
        "DIVERSITY_SCORE(col_1)",
        "MEAN_CHARACTERS_PER_WORD(col_1)",
        "NUM_CHARACTERS(col_1)",
        "NUM_WORDS(col_1)",
        "POLARITY_SCORE(col_1)",
    ]
    # find the columns that should be null
    assert all(X_t[cols].iloc[4, :].isnull())
    # these columns should not have any null values
    assert not X_t[cols].iloc[:4, :].isnull().any().any()
    assert not X_t[X_t.columns.difference(cols)].isnull().any().any()


@pytest.mark.parametrize("nones", [np.nan, pd.NA, None])
def test_multiple_nan_allowed(nones):
    X = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "",
                None,
            ],
            "col_2": [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living Red, the blood of angry men - black, the dark of ages past",
                ":)",
                nones,
            ],
            "col_3": [1, 2, 3, 1],
        },
    )
    X.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    tf = NaturalLanguageFeaturizer()
    tf.fit(X)
    X_t = tf.transform(X)
    col_names = [
        "LSA({})[0]",
        "LSA({})[1]",
        "DIVERSITY_SCORE({})",
        "MEAN_CHARACTERS_PER_WORD({})",
        "NUM_CHARACTERS({})",
        "NUM_WORDS({})",
        "POLARITY_SCORE({})",
    ]
    cols = [c.format(n) for n in ["col_1", "col_2"] for c in col_names]
    # find the columns that should be null
    assert all(X_t[cols].iloc[3, :].isnull())
    # these columns should not have any null values
    assert not X_t[cols].iloc[:3, :].isnull().any().any()
    assert not X_t[X_t.columns.difference(cols)].isnull().any().any()
    assert all([pd.api.types.is_numeric_dtype(types) for types in X_t[cols].dtypes])
