import numpy as np
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines import KModesClusterer
from evalml.problem_types import ProblemTypes

pytestmark = pytest.mark.noncore_dependency


def test_model_family():
    assert KModesClusterer.model_family == ModelFamily.CENTROID


def test_problem_types():
    assert KModesClusterer.supported_problem_types == [ProblemTypes.CLUSTERING]


def test_fit_predict(X_y_regression, kmodes):
    X, y = X_y_regression

    sk_model = kmodes.KModes(n_clusters=8, max_iter=300, random_state=0)
    sk_model.fit(X, y)
    y_pred_sk = sk_model.labels_

    model = KModesClusterer()
    fitted = model.fit(X, y)
    assert isinstance(fitted, KModesClusterer)

    y_pred = model.predict(X)
    np.testing.assert_almost_equal(y_pred_sk, y_pred.values, decimal=5)

    assert len(set(y_pred)) > 1
    assert max(y_pred) > 0
