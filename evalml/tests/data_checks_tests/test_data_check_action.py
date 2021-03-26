from evalml.data_checks import DataCheckAction, DataCheckActionCode


def test_data_check_action_attributes():
    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL)
    assert data_check_action.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action.metadata == {}

    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, {})
    assert data_check_action.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action.metadata == {}

    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, metadata={"columns": [1, 2]})
    assert data_check_action.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action.metadata == {"columns": [1, 2]}


def test_data_check_action_equality():
    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL)
    data_check_action_eq = DataCheckAction(DataCheckActionCode.DROP_COL)

    assert data_check_action == data_check_action
    assert data_check_action == data_check_action_eq
    assert data_check_action_eq == data_check_action

    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, metadata={'same detail': 'same same same'})
    data_check_action_eq = DataCheckAction(DataCheckActionCode.DROP_COL, metadata={'same detail': 'same same same'})

    assert data_check_action == data_check_action
    assert data_check_action == data_check_action_eq
    assert data_check_action_eq == data_check_action


def test_data_check_action_inequality():
    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL)
    data_check_action_diff = DataCheckAction(DataCheckActionCode.DROP_COL, metadata={"metadata": ["this is different"]})

    assert data_check_action != data_check_action_diff
    assert data_check_action_diff != data_check_action


def test_data_check_action_to_dict():
    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL)
    data_check_action_empty_metadata = DataCheckAction(DataCheckActionCode.DROP_COL, metadata={})
    data_check_action_with_metadata = DataCheckAction(DataCheckActionCode.DROP_COL, metadata={"some detail": ["this is different"]})

    assert data_check_action.to_dict() == {"code": DataCheckActionCode.DROP_COL.name, "metadata": {}}
    assert data_check_action_empty_metadata.to_dict() == {"code": DataCheckActionCode.DROP_COL.name, "metadata": {}}
    assert data_check_action_with_metadata.to_dict() == {"code": DataCheckActionCode.DROP_COL.name, "metadata": {"some detail": ["this is different"]}}
