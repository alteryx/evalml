from evalml.objectives import PrecisionMicro
from evalml.pipelines import CatBoostClassificationPipeline


def test_catboost_init():
    objective = PrecisionMicro()
    clf = CatBoostClassificationPipeline(objective=objective, impute_strategy='mean')
    expected_parameters = {'impute_strategy': 'mean'}
    assert clf.parameters == expected_parameters
    assert clf.random_state == 0

