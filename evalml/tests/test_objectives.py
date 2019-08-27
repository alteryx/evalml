import evalml.objectives.utils as objective_utils
from evalml.objectives import Precision

def test_get_objective():
    assert isinstance(objective_utils.get_objective('precision'), Precision)

def test_get_objectives_types():
    assert len(objective_utils.get_objectives('multiclass')) == 14
    assert len(objective_utils.get_objectives('binary')) == 6
    assert len(objective_utils.get_objectives('regression')) == 1