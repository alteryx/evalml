import pandas as pd
import pytest
import woodwork as ww

from evalml.pipelines.components import EmailFeaturizer, URLFeaturizer


@pytest.mark.parametrize(
    "component_class,params",
    [(URLFeaturizer, {}), (EmailFeaturizer, {})],
)
def test_init(component_class, params):
    assert component_class().parameters == params


def make_data_email_fit_transform(df_with_url_and_email):
    return df_with_url_and_email


def make_data_url_fit_transform(df_with_url_and_email):
    return df_with_url_and_email


def make_data_email_fit_transform_missing_values(df_with_url_and_email):
    df_with_missing_values = df_with_url_and_email.ww.copy()

    original_ltypes = df_with_url_and_email.ww.schema.logical_types
    df_with_missing_values.email.iloc[0:2] = pd.NA
    df_with_missing_values.ww["email_2"] = df_with_missing_values.email
    df_with_missing_values.ww["email_2"].iloc[-1] = pd.NA
    original_ltypes.update({"email_2": "EmailAddress"})
    df_with_missing_values.ww.init(logical_types=original_ltypes)
    return df_with_missing_values


def make_data_url_fit_transform_missing_values(df_with_url_and_email):
    df_with_missing_values = df_with_url_and_email.ww.copy()
    original_ltypes = df_with_url_and_email.ww.schema.logical_types
    df_with_missing_values.url.iloc[0:2] = pd.NA
    df_with_missing_values.ww["url_2"] = df_with_missing_values.url
    df_with_missing_values.ww["url_2"].iloc[-1] = pd.NA
    original_ltypes.update({"url_2": "URL"})
    df_with_missing_values.ww.init(logical_types=original_ltypes)
    return df_with_missing_values


def make_answer_email_fit_transform(df_with_url_and_email):
    expected = df_with_url_and_email.ww.copy()
    expected.ww["EMAIL_ADDRESS_TO_DOMAIN(email)"] = pd.Series(
        ["gmail.com", "yahoo.com", "abalone.com", "hotmail.com", "email.org"],
        dtype="category",
    )
    expected.ww["IS_FREE_EMAIL_DOMAIN(email)"] = ww.init_series(
        pd.Series(
            [True, True, False, True, True],
        ),
        logical_type="BooleanNullable",
    )
    expected.ww.drop(["email"], inplace=True)
    return expected


def make_answer_url_fit_transform(df_with_url_and_email):
    expected = df_with_url_and_email.ww.copy()
    expected.ww["URL_TO_DOMAIN(url)"] = pd.Series(
        [
            "evalml.alteryx.com",
            "woodwork.alteryx.com",
            "twitter.com",
            "twitter.com",
            "evalml.alteryx.com",
        ],
        dtype="category",
    )
    expected.ww["URL_TO_PROTOCOL(url)"] = pd.Series(["https"] * 5, dtype="category")
    expected.ww["URL_TO_TLD(url)"] = pd.Series(["com"] * 5, dtype="category")
    expected.ww.drop(["url"], inplace=True)
    return expected


def make_answer_email_fit_transform_missing_values(df_with_url_and_email):
    df_with_missing_values = make_data_email_fit_transform_missing_values(
        df_with_url_and_email,
    )
    expected = df_with_missing_values.ww.copy()
    expected.ww.drop(["email", "email_2"], inplace=True)
    # Missing values in the original features are passed through
    expected.ww["EMAIL_ADDRESS_TO_DOMAIN(email)"] = pd.Series(
        [None, None, "abalone.com", "hotmail.com", "email.org"],
        dtype="category",
    )
    expected.ww["EMAIL_ADDRESS_TO_DOMAIN(email_2)"] = pd.Series(
        [None, None, "abalone.com", "hotmail.com", None],
        dtype="category",
    )
    expected.ww["IS_FREE_EMAIL_DOMAIN(email)"] = pd.Series(
        [None, None, False, True, True],
        dtype="boolean",
    )
    expected.ww["IS_FREE_EMAIL_DOMAIN(email_2)"] = pd.Series(
        [None, None, False, True, None],
        dtype="boolean",
    )
    return expected


def make_answer_url_fit_transform_missing_values(df_with_url_and_email):
    df_with_missing_values = make_data_url_fit_transform_missing_values(
        df_with_url_and_email,
    )
    expected = df_with_missing_values.ww.copy()
    expected.ww.drop(["url", "url_2"], inplace=True)
    # Missing values in the original features are passed through
    expected.ww["URL_TO_DOMAIN(url)"] = pd.Series(
        [None, None, "twitter.com", "twitter.com", "evalml.alteryx.com"],
        dtype="category",
    )
    expected.ww["URL_TO_DOMAIN(url_2)"] = pd.Series(
        [None, None, "twitter.com", "twitter.com", None],
        dtype="category",
    )
    expected.ww["URL_TO_PROTOCOL(url)"] = pd.Series(
        [None, None] + ["https"] * 3,
        dtype="category",
    )
    expected.ww["URL_TO_PROTOCOL(url_2)"] = pd.Series(
        [None, None] + ["https"] * 2 + [None],
        dtype="category",
    )
    expected.ww["URL_TO_TLD(url)"] = pd.Series(
        [None, None] + ["com"] * 3,
        dtype="category",
    )

    expected.ww["URL_TO_TLD(url_2)"] = pd.Series(
        [None, None] + ["com"] * 2 + [None],
        dtype="category",
    )
    return expected


def make_expected_logical_types_email_fit_transform():
    return {
        "categorical": ww.logical_types.Categorical(),
        "numeric": ww.logical_types.Double(),
        "IS_FREE_EMAIL_DOMAIN(email)": ww.logical_types.BooleanNullable(),
        "EMAIL_ADDRESS_TO_DOMAIN(email)": ww.logical_types.Categorical(),
        "integer": ww.logical_types.Integer(),
        "boolean": ww.logical_types.Boolean(),
        "nat_lang": ww.logical_types.NaturalLanguage(),
        "url": ww.logical_types.URL(),
    }


def make_expected_logical_types_url_fit_transform():
    return {
        "categorical": ww.logical_types.Categorical(),
        "numeric": ww.logical_types.Double(),
        "email": ww.logical_types.EmailAddress(),
        "integer": ww.logical_types.Integer(),
        "boolean": ww.logical_types.Boolean(),
        "nat_lang": ww.logical_types.NaturalLanguage(),
        "URL_TO_DOMAIN(url)": ww.logical_types.Categorical(),
        "URL_TO_PROTOCOL(url)": ww.logical_types.Categorical(),
        "URL_TO_TLD(url)": ww.logical_types.Categorical(),
    }


def make_expected_logical_types_email_fit_transform_missing_values():
    return {
        "categorical": ww.logical_types.Categorical(),
        "numeric": ww.logical_types.Double(),
        "EMAIL_ADDRESS_TO_DOMAIN(email)": ww.logical_types.Categorical(),
        "EMAIL_ADDRESS_TO_DOMAIN(email_2)": ww.logical_types.Categorical(),
        "IS_FREE_EMAIL_DOMAIN(email)": ww.logical_types.BooleanNullable(),
        "IS_FREE_EMAIL_DOMAIN(email_2)": ww.logical_types.BooleanNullable(),
        "integer": ww.logical_types.Integer(),
        "boolean": ww.logical_types.Boolean(),
        "nat_lang": ww.logical_types.NaturalLanguage(),
        "url": ww.logical_types.URL(),
    }


def make_expected_logical_types_url_fit_transform_missing_values():
    return {
        "categorical": ww.logical_types.Categorical(),
        "numeric": ww.logical_types.Double(),
        "email": ww.logical_types.EmailAddress(),
        "integer": ww.logical_types.Integer(),
        "boolean": ww.logical_types.Boolean(),
        "nat_lang": ww.logical_types.NaturalLanguage(),
        "URL_TO_DOMAIN(url)": ww.logical_types.Categorical(),
        "URL_TO_PROTOCOL(url)": ww.logical_types.Categorical(),
        "URL_TO_TLD(url)": ww.logical_types.Categorical(),
        "URL_TO_DOMAIN(url_2)": ww.logical_types.Categorical(),
        "URL_TO_PROTOCOL(url_2)": ww.logical_types.Categorical(),
        "URL_TO_TLD(url_2)": ww.logical_types.Categorical(),
    }


@pytest.mark.parametrize(
    "component, make_data, make_expected, make_expected_ltypes",
    [
        (
            EmailFeaturizer(),
            make_data_email_fit_transform,
            make_answer_email_fit_transform,
            make_expected_logical_types_email_fit_transform,
        ),
        (
            EmailFeaturizer(),
            make_data_email_fit_transform_missing_values,
            make_answer_email_fit_transform_missing_values,
            make_expected_logical_types_email_fit_transform_missing_values,
        ),
        (
            URLFeaturizer(),
            make_data_url_fit_transform,
            make_answer_url_fit_transform,
            make_expected_logical_types_url_fit_transform,
        ),
        (
            URLFeaturizer(),
            make_data_url_fit_transform_missing_values,
            make_answer_url_fit_transform_missing_values,
            make_expected_logical_types_url_fit_transform_missing_values,
        ),
    ],
)
def test_component_fit_transform(
    component,
    make_data,
    make_expected,
    make_expected_ltypes,
    df_with_url_and_email,
):
    data = make_data(df_with_url_and_email)
    expected = make_expected(data)
    expected_logical_types = make_expected_ltypes()

    component.fit(data)
    new_X = component.transform(data)

    # TODO: Understand why the "EMAIL_ADDRESS_TO_DOMAIN(email)" engineered feature
    # returns categories of type "object" when making the expected answer but
    # returns categories of type "string" when run through featuretools after
    # moving to pandas 1.4.0.  Revert this change from GH #3324 when this is understood.

    # pd.testing.assert_frame_equal(new_X, expected) <- this should work
    assert all(new_X == expected)
    assert all(new_X.columns == expected.columns)
    assert all(new_X.index == expected.index)
    assert new_X.ww.logical_types == expected.ww.logical_types

    assert new_X.ww.logical_types == expected_logical_types

    new_X = component.fit_transform(data)

    # pd.testing.assert_frame_equal(new_X, expected)
    assert all(new_X == expected)
    assert all(new_X.columns == expected.columns)
    assert all(new_X.index == expected.index)
    assert new_X.ww.logical_types == expected.ww.logical_types
