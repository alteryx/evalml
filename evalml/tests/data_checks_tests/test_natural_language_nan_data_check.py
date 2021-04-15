import numpy as np
import pandas as pd
import woodwork as ww

from evalml.data_checks import (
    DataCheckError,
    DataCheckMessageCode,
    NaturalLanguageNaNDataCheck
)


def test_nl_nan_data_check_error():
    data = pd.DataFrame({'natural_language': [None, "string_that_is_long_enough_for_natural_language", "string_that_is_long_enough_for_natural_language"]})
    nl_nan_check = NaturalLanguageNaNDataCheck()
    assert nl_nan_check.validate(data) == {
        "warnings": [],
        "actions": [],
        "errors": [DataCheckError(message='Input natural language column(s) (natural_language) contains NaN values. Please impute NaN values or drop these rows or columns.',
                                  data_check_name=NaturalLanguageNaNDataCheck.name,
                                  message_code=DataCheckMessageCode.NATURAL_LANGUAGE_HAS_NAN,
                                  details={"columns": 'natural_language'}).to_dict()]
    }


def test_nl_nan_data_check_error_no_nan():
    nl_nan_check = NaturalLanguageNaNDataCheck()
    assert nl_nan_check.validate(pd.DataFrame({'natural_language': ["string_that_is_long_enough_for_natural_language", "string_that_is_long_enough_for_natural_language"]})) == {
        "warnings": [],
        "actions": [],
        "errors": []
    }


def test_nl_nan_data_check_error_other_cols_with_nan():
    data = pd.DataFrame(np.random.randint(0, 10, size=(2, 2)))
    data['A'] = ['string_that_is_long_enough_for_natural_language', 'string_that_is_long_enough_for_natural_language']
    data = data.replace(data.iloc[0][0], None)
    data = data.replace(data.iloc[1][1], None)
    nl_nan_check = NaturalLanguageNaNDataCheck()
    assert nl_nan_check.validate(data) == {
        "warnings": [],
        "actions": [],
        "errors": []
    }


def test_nl_nan_data_check_error_multiple_nl_no_nan():
    data = pd.DataFrame()
    data['A'] = ['string_that_is_long_enough_for_natural_language', 'string_that_is_long_enough_for_natural_language']
    data['B'] = ['string_that_is_long_enough_for_natural_language', 'string_that_is_long_enough_for_natural_language']

    data['C'] = np.random.randint(0, 3, size=len(data))

    nl_nan_check = NaturalLanguageNaNDataCheck()
    assert nl_nan_check.validate(data) == {
        "warnings": [],
        "actions": [],
        "errors": []
    }


def test_nl_nan_data_check_error_multiple_nl_nan():
    data = pd.DataFrame()
    data['A'] = pd.Series([None, "string_that_is_long_enough_for_natural_language", "string_that_is_long_enough_for_natural_language"])
    data['B'] = pd.Series([None, "string_that_is_long_enough_for_natural_language", "string_that_is_long_enough_for_natural_language"])
    data['C'] = pd.Series(["", "string_that_is_long_enough_for_natural_language", "string_that_is_long_enough_for_natural_language"])
    data['D'] = np.random.randint(0, 5, size=len(data))

    nl_nan_check = NaturalLanguageNaNDataCheck()
    assert nl_nan_check.validate(data) == {
        "warnings": [],
        "actions": [],
        "errors": [DataCheckError(message='Input natural language column(s) (A, B) contains NaN values. Please impute NaN values or drop these rows or columns.',
                                  data_check_name=NaturalLanguageNaNDataCheck.name,
                                  message_code=DataCheckMessageCode.NATURAL_LANGUAGE_HAS_NAN,
                                  details={"columns": 'A, B'}).to_dict()]
    }


def test_nl_nan_check_input_formats():
    nl_nan_check = NaturalLanguageNaNDataCheck()

    # test empty pd.DataFrame
    assert nl_nan_check.validate(pd.DataFrame()) == {"warnings": [], "errors": [], "actions": []}

    expected = {
        "warnings": [],
        "actions": [],
        "errors": [DataCheckError(message='Input natural language column(s) (nl) contains NaN values. Please impute NaN values or drop these rows or columns.',
                                  data_check_name=NaturalLanguageNaNDataCheck.name,
                                  message_code=DataCheckMessageCode.NATURAL_LANGUAGE_HAS_NAN,
                                  details={"columns": 'nl'}).to_dict()]
    }

    nl_col = [None, "string_that_is_long_enough_for_natural_language", "string_that_is_long_enough_for_natural_language"]

    #  test Woodwork
    ww_input = ww.DataTable(pd.DataFrame(nl_col, columns=['nl']), logical_types={'nl': 'NaturalLanguage'})
    assert nl_nan_check.validate(ww_input) == expected

    expected = {
        "warnings": [],
        "actions": [],
        "errors": [DataCheckError(message='Input natural language column(s) (0) contains NaN values. Please impute NaN values or drop these rows or columns.',
                                  data_check_name=NaturalLanguageNaNDataCheck.name,
                                  message_code=DataCheckMessageCode.NATURAL_LANGUAGE_HAS_NAN,
                                  details={'columns': '0'}).to_dict()]
    }

    #  test 2D list
    nl_col_without_nan = ["string_that_is_long_enough_for_natural_language", "string_that_is_long_enough_for_natural_language", "string_that_is_long_enough_for_natural_language"]
    assert nl_nan_check.validate([nl_col, nl_col_without_nan]) == expected

    # test np.array
    assert nl_nan_check.validate(np.array([nl_col, nl_col_without_nan])) == expected
