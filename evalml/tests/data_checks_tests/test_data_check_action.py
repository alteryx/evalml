import pytest

from evalml.data_checks import DataCheckAction, DataCheckActionCode


def test_data_check_action_attributes(dummy_data_check_name):
    data_check_action = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
    )
    assert data_check_action.data_check_name == dummy_data_check_name
    assert data_check_action.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action.metadata == {"rows": None, "columns": None}

    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, None, metadata={})
    assert data_check_action.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action.data_check_name is None
    assert data_check_action.metadata == {"rows": None, "columns": None}

    data_check_action = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={"columns": [1, 2]},
    )
    assert data_check_action.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action.data_check_name == dummy_data_check_name
    assert data_check_action.metadata == {"columns": [1, 2], "rows": None}


def test_data_check_action_equality(dummy_data_check_name):
    data_check_action = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
    )
    data_check_action_eq = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
    )

    assert data_check_action == data_check_action
    assert data_check_action == data_check_action_eq
    assert data_check_action_eq == data_check_action

    data_check_action = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        None,
        metadata={"same detail": "same same same"},
    )
    data_check_action_eq = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        None,
        metadata={"same detail": "same same same"},
    )

    assert data_check_action == data_check_action
    assert data_check_action == data_check_action_eq
    assert data_check_action_eq == data_check_action


def test_data_check_action_inequality():
    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, None)
    data_check_action_diff = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        None,
        metadata={"metadata": ["this is different"]},
    )

    assert data_check_action != data_check_action_diff
    assert data_check_action_diff != data_check_action


def test_data_check_action_to_dict(dummy_data_check_name):
    data_check_action = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
    )
    data_check_action_empty_metadata = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={},
    )
    data_check_action_with_metadata = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={"some detail": ["this is different"]},
    )

    assert data_check_action.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "metadata": {"columns": None, "rows": None},
    }
    assert data_check_action_empty_metadata.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "metadata": {"columns": None, "rows": None},
    }
    assert data_check_action_with_metadata.to_dict() == {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "metadata": {
            "some detail": ["this is different"],
            "columns": None,
            "rows": None,
        },
    }


def test_convert_dict_to_action_bad_input():
    data_check_action_dict_no_code = {
        "metadata": {"columns": None, "rows": None},
    }
    with pytest.raises(ValueError, match="The input dictionary should have the keys"):
        DataCheckAction.convert_dict_to_action(data_check_action_dict_no_code)

    data_check_action_dict_no_metadata = {
        "code": DataCheckActionCode.DROP_COL.name,
    }
    with pytest.raises(ValueError, match="The input dictionary should have the keys"):
        DataCheckAction.convert_dict_to_action(data_check_action_dict_no_metadata)

    data_check_action_dict_no_columns = {
        "code": DataCheckActionCode.DROP_COL.name,
        "metadata": {"cow": None},
    }
    with pytest.raises(
        ValueError,
        match="The metadata dictionary should have the keys",
    ):
        DataCheckAction.convert_dict_to_action(data_check_action_dict_no_columns)


def test_convert_dict_to_action(dummy_data_check_name):
    data_check_action_dict = {
        "code": DataCheckActionCode.DROP_COL.name,
        "metadata": {"columns": None, "rows": None},
    }
    expected_data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, None)
    data_check_action = DataCheckAction.convert_dict_to_action(data_check_action_dict)
    assert data_check_action == expected_data_check_action

    data_check_action_dict_with_other_metadata = {
        "code": DataCheckActionCode.DROP_COL.name,
        "data_check_name": dummy_data_check_name,
        "metadata": {
            "some detail": ["this is different"],
            "columns": None,
            "rows": None,
        },
    }
    expected_data_check_action = DataCheckAction(
        DataCheckActionCode.DROP_COL,
        dummy_data_check_name,
        metadata={"some detail": ["this is different"]},
    )
    data_check_action = DataCheckAction.convert_dict_to_action(
        data_check_action_dict_with_other_metadata,
    )
    assert data_check_action == expected_data_check_action


@pytest.mark.parametrize(
    "action_code,expected_code",
    [
        ("drop_rows", DataCheckActionCode.DROP_ROWS),
        ("Drop_col", DataCheckActionCode.DROP_COL),
        ("TRANSFORM_TARGET", DataCheckActionCode.TRANSFORM_TARGET),
    ],
)
def test_data_check_action_equality_string_input(
    action_code,
    expected_code,
    dummy_data_check_name,
):
    data_check_action = DataCheckAction(action_code, dummy_data_check_name)
    data_check_action_eq = DataCheckAction(expected_code, dummy_data_check_name)

    assert data_check_action == data_check_action
    assert data_check_action == data_check_action_eq
    assert data_check_action_eq == data_check_action

    data_check_action = DataCheckAction(
        action_code,
        None,
        metadata={"same detail": "same same same"},
    )
    data_check_action_eq = DataCheckAction(
        expected_code,
        None,
        metadata={"same detail": "same same same"},
    )

    assert data_check_action == data_check_action
    assert data_check_action == data_check_action_eq
    assert data_check_action.to_dict() == data_check_action_eq.to_dict()
