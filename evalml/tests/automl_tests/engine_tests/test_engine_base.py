import pandas as pd
import pytest

from evalml.automl.automl_search import AutoMLSearch
from evalml.automl.engines import EngineBase


class DummyEngine(EngineBase):
    def __init__(self):
        super().__init__()
        self.name = 'Test Dummy Engine'

    def evaluate_batch(self, pipeline_batch):
        super().evaluate_batch()

    def evaluate_pipeline(self, pipeline):
        super().evaluate_pipeline()


def test_load_data(X_y_binary):
    engine = DummyEngine()
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    engine.load_data(X, y)
    pd.testing.assert_frame_equal(X, engine.X_train)
    pd.testing.assert_frame_equal(y, engine.y_train)


def test_load_automl(X_y_binary):
    engine = DummyEngine()
    X, y = X_y_binary
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary')
    engine.load_search(automl)
    assert engine.automl == automl


def test_load_data_before_evaluate():
    engine = DummyEngine()
    expected_error = "Dataset has not been loaded into the engine. Call `load_data` with training data."
    with pytest.raises(ValueError, match=expected_error):
        engine.evaluate_batch([])


def test_load_search_before_evaluate(X_y_binary):
    engine = DummyEngine()
    X, y = X_y_binary
    engine.load_data(X, y)
    expected_error = "Search info has not been loaded into the engine. Call `load_search` with search context."
    with pytest.raises(ValueError, match=expected_error):
        engine.evaluate_batch([])
