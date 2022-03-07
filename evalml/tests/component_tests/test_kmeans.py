import numpy as np
from sklearn.cluster import KMeans as SKKMeans

from evalml.model_family import ModelFamily
from evalml.pipelines import KMeansClusterer
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert KMeansClusterer.model_family == ModelFamily.CENTROID


def test_problem_types():
    assert KMeansClusterer.supported_problem_types == [ProblemTypes.CLUSTERING]


def test_fit_predict(X_y_regression):
    X, y = X_y_regression

    sk_model = SKKMeans(n_clusters=8, max_iter=300, random_state=0)
    sk_model.fit(X, y)
    y_pred_sk = sk_model.labels_

    model = KMeansClusterer()
    fitted = model.fit(X, y)
    assert isinstance(fitted, KMeansClusterer)

    y_pred = model.predict(X)
    np.testing.assert_almost_equal(y_pred_sk, y_pred.values, decimal=5)

    assert len(set(y_pred)) > 1
    assert max(y_pred) > 0
