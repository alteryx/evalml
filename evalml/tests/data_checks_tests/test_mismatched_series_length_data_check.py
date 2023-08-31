import pytest

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    MismatchedSeriesLengthDataCheck,
)

mismatch_series_length_dc_name = MismatchedSeriesLengthDataCheck.name


def test_mismatched_series_length_data_check_raises_value_error(
    multiseries_ts_data_stacked,
):
    with pytest.raises(
        ValueError,
        match="series_id must be set to the series_id column in the dataset and not None",
    ):
        MismatchedSeriesLengthDataCheck(None)

    X, _ = multiseries_ts_data_stacked
    dc = MismatchedSeriesLengthDataCheck("not_series_id")
    with pytest.raises(
        ValueError,
        match="""series_id "not_series_id" doesn't match the series_id column of the dataset.""",
    ):
        dc.validate(X)


@pytest.mark.parametrize(
    "num_drop, not_majority, majority_length",
    [(1, ["0"], 20), (2, ["0", "1"], 20), (3, ["3", "4"], 19)],
)
def test_mismatched_series_length_data_check(
    multiseries_ts_data_stacked,
    num_drop,
    not_majority,
    majority_length,
):
    X, _ = multiseries_ts_data_stacked
    for i in range(num_drop):
        X = X.drop(labels=0, axis=0).reset_index(drop=True)
    mismatch_series_length_dc = MismatchedSeriesLengthDataCheck("series_id")
    messages = mismatch_series_length_dc.validate(X)
    assert len(messages) == 1
    assert messages == [
        DataCheckWarning(
            message=f"Series ID {not_majority} do not match the majority length of the other series, which is {majority_length}",
            data_check_name=mismatch_series_length_dc_name,
            message_code=DataCheckMessageCode.MISMATCHED_SERIES_LENGTH,
            details={"series_id": not_majority, "majority_length": majority_length},
            action_options=[],
        ).to_dict(),
    ]


def test_mismatched_series_length_data_check_all(multiseries_ts_data_stacked):
    rows_index = [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]
    X, _ = multiseries_ts_data_stacked
    X = X.drop(rows_index).reset_index(drop=True)
    mismatch_series_length_dc = MismatchedSeriesLengthDataCheck("series_id")
    messages = mismatch_series_length_dc.validate(X)
    assert len(messages) == 1
    assert messages == [
        DataCheckWarning(
            message="All series ID have different lengths than each other",
            data_check_name=mismatch_series_length_dc_name,
            message_code=DataCheckMessageCode.MISMATCHED_SERIES_LENGTH,
            action_options=[],
        ).to_dict(),
    ]
