import numpy as np
import pytest
from sklearn.model_selection import KFold as sk_kfold
from sklearn.model_selection import StratifiedKFold as sk_stratified

from evalml.preprocessing.data_splitters import KFold, StratifiedKFold


@pytest.mark.parametrize(
    "sk_splitter,splitter",
    [[sk_kfold, KFold], [sk_stratified, StratifiedKFold]],
)
@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_splitters_equal(problem_type, sk_splitter, splitter, X_y_binary, X_y_multi):
    parameters = {"shuffle": True, "random_state": 0, "n_splits": 4}
    sk_split = splitter(**parameters)
    evalml_split = splitter(**parameters)
    if problem_type == "binary":
        X, y = X_y_binary
    else:
        X, y = X_y_multi

    skt, skv = [], []
    evt, evv = [], []

    for t, v in sk_split.split(X, y):
        skt.append(t)
        skv.append(v)
    for t, v in evalml_split.split(X, y):
        evt.append(t)
        evv.append(v)
    np.testing.assert_array_equal(skt, evt)
    np.testing.assert_array_equal(skv, evv)
    assert evalml_split.is_cv
