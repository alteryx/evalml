import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.model_family import ModelFamily
from evalml.pipelines.components import BaselineClusterer
from evalml.utils import get_random_state


def test_baseline_init():
    baseline = BaselineClusterer()
    assert baseline.model_family == ModelFamily.BASELINE
    assert baseline.parameters["n_clusters"] == 8


def test_baseline_invalid_n_clusters():
    with pytest.raises(ValueError, match="The number of clusters must be a whole number greater than 1."):
        BaselineClusterer(n_clusters=1)
    with pytest.raises(ValueError, match="The number of clusters must be a whole number greater than 1."):
        BaselineClusterer(n_clusters="three")


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_baseline_n_classes(data_type, make_data_type):
    X = pd.DataFrame({"one": [1, 2, 3, 4]*10, "two": [2, 3, 4, 5]*10, "three": [1, 2, 3, 4]*10})
    X = make_data_type(data_type, X)

    clf = BaselineClusterer(n_clusters=2)
    fitted = clf.fit(X)
    assert isinstance(fitted, BaselineClusterer)
    predictions = clf.predict(X)
    assert len(np.unique(predictions)) == 2

    clf = BaselineClusterer(n_clusters=9)
    fitted = clf.fit(X)
    assert isinstance(fitted, BaselineClusterer)
    predictions = clf.predict(X)
    assert len(np.unique(predictions)) == 9

    np.testing.assert_allclose(clf.feature_importance, np.array([0.0] * X.shape[1]))
