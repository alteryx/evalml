import pandas as pd

import evalml.objectives.utils as objective_utils
from evalml.objectives import Precision, PrecisionMacro, PrecisionMicro
from evalml.pipelines import LogisticRegressionPipeline


def test_get_objective():
    assert isinstance(objective_utils.get_objective('precision'), Precision)


def test_get_objectives_types():
    assert len(objective_utils.get_objectives('multiclass')) == 14
    assert len(objective_utils.get_objectives('binary')) == 6
    assert len(objective_utils.get_objectives('regression')) == 1


def test_binary_average(X_y):
    X, y = X_y
    X = pd.DataFrame(X)
    y = pd.Series(y)

    pipeline = LogisticRegressionPipeline(objective=Precision(), penalty='l2', C=1.0, impute_strategy='mean', number_features=0)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    assert Precision().score(y, y_pred) == PrecisionMicro().score(y, y_pred)
    assert Precision().score(y, y_pred) == PrecisionMacro().score(y, y_pred)
