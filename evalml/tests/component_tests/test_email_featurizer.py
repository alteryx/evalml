from evalml.pipelines.components import EmailFeaturizer
import pandas as pd
import woodwork as ww
import pytest


@pytest.fixture
def df_with_email_features():
    X = pd.DataFrame({"categorical": ["a", "b", "b", "a", "c"],
                      "numeric": [1, 2, 3, 4, 5],
                      "email": ["abalone_0@gmail.com", "AbaloneRings@yahoo.com", "abalone_2@abalone.com",
                                "$titanic_data%&@hotmail.com", "foo*EMAIL@email.org"],
                      "integer": [1, 2, 3, 4, 5],
                      "boolean": [True, False, True, False, False],
                      "nat_lang": ['natural', 'language', 'understanding', 'is', 'difficult'],
                      "url": ["https://evalml.alteryx.com/en/stable/",
                              "https://woodwork.alteryx.com/en/stable/guides/statistical_insights.html",
                              "https://twitter.com/AlteryxOSS",
                              "https://www.twitter.com/AlteryxOSS",
                              "www.evalml.alteryx.com/en/stable/demos/text_input.html"]})
    X.ww.init(logical_types={"categorical": "Categorical", "numeric": "Double", "email": "EmailAddress",
                             "boolean": "Boolean", 'nat_lang': "NaturalLanguage", "integer": "Integer",
                             "url": "URL"})
    return X


def test_email_featurizer_init():
    email = EmailFeaturizer()
    assert email.parameters == {}


def test_email_featurizer_fit_transform(df_with_email_features):

    email = EmailFeaturizer()
    email.fit(df_with_email_features)
    new_X = email.transform(df_with_email_features)
    expected = df_with_email_features.ww.copy()
    expected_logical_types = {"categorical": ww.logical_types.Categorical(),
                              "numeric": ww.logical_types.Double(),
                              "IS_FREE_EMAIL_DOMAIN(email)": ww.logical_types.Boolean(),
                              "integer": ww.logical_types.Integer(),
                              "boolean": ww.logical_types.Boolean(),
                              "nat_lang": ww.logical_types.NaturalLanguage(),
                              "url": ww.logical_types.URL()}
    expected.ww['IS_FREE_EMAIL_DOMAIN(email)'] = pd.Series([True, True, False, True, True])
    expected.ww.drop(['email'], inplace=True)
    pd.testing.assert_frame_equal(new_X, expected)
    assert new_X.ww.logical_types == expected_logical_types

    new_X = email.fit_transform(df_with_email_features)
    pd.testing.assert_frame_equal(new_X, expected)
    assert new_X.ww.logical_types == expected_logical_types


def test_email_featurizer_fit_transform_missing_values(df_with_email_features):
    df_with_missing_values = df_with_email_features.ww.copy()
    original_ltypes = df_with_email_features.ww.schema.logical_types
    df_with_missing_values.email.iloc[0:2] = pd.NA
    df_with_missing_values.ww['email_2'] = df_with_missing_values.email
    df_with_missing_values.ww['email_2'].iloc[-1] = pd.NA
    original_ltypes.update({"email_2": "EmailAddress"})
    df_with_missing_values.ww.init(logical_types=original_ltypes)

    email = EmailFeaturizer()
    new = email.fit_transform(df_with_missing_values)

    expected = df_with_missing_values.ww.copy()
    expected.ww.drop(['email', 'email_2'], inplace=True)
    # Missing values in the original features return False
    expected.ww['IS_FREE_EMAIL_DOMAIN(email)'] = pd.Series([False, False, False, True, True])
    expected.ww['IS_FREE_EMAIL_DOMAIN(email_2)'] = pd.Series([False, False, False, True, False])
    pd.testing.assert_frame_equal(new, expected)
    expected_logical_types = {"categorical": ww.logical_types.Categorical(),
                              "numeric": ww.logical_types.Double(),
                              "IS_FREE_EMAIL_DOMAIN(email)": ww.logical_types.Boolean(),
                              "IS_FREE_EMAIL_DOMAIN(email_2)": ww.logical_types.Boolean(),
                              "integer": ww.logical_types.Integer(),
                              "boolean": ww.logical_types.Boolean(),
                              "nat_lang": ww.logical_types.NaturalLanguage(),
                              "url": ww.logical_types.URL()}
    assert new.ww.logical_types == expected_logical_types


