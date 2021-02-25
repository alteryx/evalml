from evalml.data_checks import DataCheckAction, DataCheckActionCode


def test_data_check_action_attributes():
    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, {})
    assert data_check_action.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action.details == {}

    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, {"cols": [1, 2]})
    assert data_check_action.action_code == DataCheckActionCode.DROP_COL
    assert data_check_action.details == {"cols": [1, 2]}


def test_data_check_action_equality():
    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, {})
    data_check_action_eq = DataCheckAction(DataCheckActionCode.DROP_COL, {})

    assert data_check_action == data_check_action
    assert data_check_action == data_check_action_eq
    assert data_check_action_eq == data_check_action


def test_data_check_action_inequality():
    data_check_action = DataCheckAction(DataCheckActionCode.DROP_COL, {})
    data_check_action_diff = DataCheckAction(DataCheckActionCode.DROP_COL, {"details": ["this is different"]})

    assert data_check_action != data_check_action_diff
    assert data_check_action_diff != data_check_action
