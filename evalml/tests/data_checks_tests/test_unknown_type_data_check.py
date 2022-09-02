import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    UnknownTypeDataCheck,
)

unknown_type_data_check_name = UnknownTypeDataCheck.name


@pytest.fixture
def test_unknown_data_type_check_init():
    unknown_type_data_check = UnknownTypeDataCheck()
    assert unknown_type_data_check.unknown_percentage_threshold == 0.50

    unknown_type_data_check = UnknownTypeDataCheck(unknown_percentage_threshold=0.60)
    assert unknown_type_data_check.unknown_percentage_threshold == 0.6

    unknown_type_data_check = UnknownTypeDataCheck(unknown_percentage_threshold=1)
    assert unknown_type_data_check.unknown_percentage_threshold == 1

    with pytest.raises(
        ValueError,
        match="`unknown_percentage_threshold` must be a float between 0 and 1, inclusive.",
    ):
        UnknownTypeDataCheck(unknown_percentage_threshold=1.1)
    with pytest.raises(
        ValueError,
        match="`unknown_percentage_threshold` must be a float between 0 and 1, inclusive.",
    ):
        UnknownTypeDataCheck(unknown_percentage_threshold=-0.1)


def test_unknown_data_type_check_warnings():
    dataframe = pd.DataFrame(
        {
            "some_null": [2, 4, None, None, 5],
            "lots_of_null": [None, None, None, None, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        },
    )
    unknown_type_dc = UnknownTypeDataCheck()
    validate_message = unknown_type_dc.validate(dataframe)
    assert validate_message == []

    fifty_percent_unknown_type_dataframe = pd.DataFrame(
        {
            "some_null": [2, 4, None, None, 5],
            "literally_all_null": [None, None, None, None, None],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        },
    )
    validate_message = unknown_type_dc.validate(fifty_percent_unknown_type_dataframe)
    assert validate_message == [
        DataCheckWarning(
            message="2 out of 4 rows are unknown type, meaning the number of rows that are unknown is more or equal to 50.0%.",
            data_check_name=unknown_type_data_check_name,
            message_code=DataCheckMessageCode.HIGH_NUMBER_OF_UNKNOWN_TYPE,
            details={
                "columns": ["literally_all_null", "all_null"],
            },
            action_options=[],
        ).to_dict(),
    ]

    one_hundred_percent_unknown_type_dataframe = pd.DataFrame(
        {
            "null": [None, None, None, None, None],
            "literally_all_null": [None, None, None, None, None],
            "all_null": [None, None, None, None, None],
            "nulls_everywhere": [None, None, None, None, None],
        },
    )
    validate_message = unknown_type_dc.validate(
        one_hundred_percent_unknown_type_dataframe,
    )
    assert validate_message == [
        DataCheckWarning(
            message="4 out of 4 rows are unknown type, meaning the number of rows that are unknown is more or equal to 50.0%.",
            data_check_name=unknown_type_data_check_name,
            message_code=DataCheckMessageCode.HIGH_NUMBER_OF_UNKNOWN_TYPE,
            details={
                "columns": [
                    "null",
                    "literally_all_null",
                    "all_null",
                    "nulls_everywhere",
                ],
            },
            action_options=[],
        ).to_dict(),
    ]
