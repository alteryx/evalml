import numpy as np
import pandas as pd
import pytest
from kmodes.kprototypes import KPrototypes as SKKPrototypes

from evalml.model_family import ModelFamily
from evalml.pipelines import KPrototypesClusterer
from evalml.problem_types import ProblemTypes

pytestmark = pytest.mark.noncore_dependency


def test_model_family():
    assert KPrototypesClusterer.model_family == ModelFamily.CENTROID


def test_problem_types():
    assert KPrototypesClusterer.supported_problem_types == [ProblemTypes.CLUSTERING]


def test_fit_predict():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", "a", "a", "b", "a", "c", "d"],
            "col_2": ["a", "b", "a", "c", "b", "a", "a", "b", "c", "a"],
            "col_3": [1, 3, 2, 4, 2, 3, 2, 1, 4, 3],
        }
    )
    X.ww.init(logical_types={"col_1": "categorical", "col_2": "categorical"})

    sk_model = SKKPrototypes(n_clusters=2, max_iter=300, random_state=0)
    sk_model.fit(X, categorical=[0, 1])
    y_pred_sk = sk_model.labels_

    model = KPrototypesClusterer(n_clusters=2)
    fitted = model.fit(X)
    assert isinstance(fitted, KPrototypesClusterer)

    y_pred = model.predict(X)
    np.testing.assert_almost_equal(y_pred_sk, y_pred.values, decimal=5)

    assert len(set(y_pred)) > 1
    assert max(y_pred) > 0
