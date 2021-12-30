import pytest

from evalml.data_checks import DataCheckActionCode
from evalml.data_checks.utils import handle_data_check_action_code
from evalml.problem_types import ProblemTypes


def test_handle_action_code_errors():
    with pytest.raises(KeyError, match="Action code 'dropping cols' does not"):
        handle_data_check_action_code("dropping cols")

    with pytest.raises(
        ValueError,
        match="`handle_data_check_action_code` was not passed a str or DataCheckActionCode object",
    ):
        handle_data_check_action_code(None)

    with pytest.raises(
        ValueError,
        match="`handle_data_check_action_code` was not passed a str or DataCheckActionCode object",
    ):
        handle_data_check_action_code(ProblemTypes.BINARY)


@pytest.mark.parametrize(
    "action_code_str,expected_enum",
    [
        ("drop_rows", DataCheckActionCode.DROP_ROWS),
        ("Drop_col", DataCheckActionCode.DROP_COL),
        ("TRANSFORM_TARGET", DataCheckActionCode.TRANSFORM_TARGET),
        (DataCheckActionCode.IMPUTE_COL, DataCheckActionCode.IMPUTE_COL),
    ],
)
def test_handle_action_code(action_code_str, expected_enum):
    assert handle_data_check_action_code(action_code_str) == expected_enum
