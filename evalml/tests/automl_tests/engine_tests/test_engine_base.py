import pandas as pd
import pytest

from evalml.automl.automl_search import AutoMLSearch
from evalml.automl.engine import EngineBase


class DummyEngine(EngineBase):
    def __init__(self):
        super().__init__()
        self.name = 'Test Dummy Engine'

    def evaluate_batch(self, pipeline_batch):
        """No-op"""


def test_set_data(X_y_binary):
    engine = DummyEngine()
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    engine.set_data(X, y)
    pd.testing.assert_frame_equal(X, engine.X_train)
    pd.testing.assert_frame_equal(y, engine.y_train)
