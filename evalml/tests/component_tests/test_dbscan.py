import numpy as np
from sklearn.cluster import DBSCAN as SKDBSCAN

from evalml.model_family import ModelFamily
from evalml.pipelines import DBSCANClusterer
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert DBSCANClusterer.model_family == ModelFamily.DENSITY


def test_problem_types():
    assert DBSCANClusterer.supported_problem_types == [ProblemTypes.CLUSTERING]


def test_fit_predict(X_y_regression):
    X, y = X_y_regression

    sk_model = SKDBSCAN(eps=4, min_samples=2)
    sk_model.fit(X, y)
    y_pred_sk = sk_model.labels_

    model = DBSCANClusterer(eps=4, min_samples=2)
    fitted = model.fit(X, y)
    assert isinstance(fitted, DBSCANClusterer)

    y_pred = model.predict(X)
    np.testing.assert_almost_equal(y_pred_sk, y_pred.values, decimal=5)

    assert len(set(y_pred)) > 1
    assert max(y_pred) > 0
