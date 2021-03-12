import pytest
from dask.distributed import Client

from evalml.pipelines import BinaryClassificationPipeline
from evalml.automl.engine.dask_engine import DaskEngine

client = Client()
automl_data =

def test_init():
    engine = DaskEngine(client=client)
    assert engine.client == client

    with pytest.raises(TypeError, match="Expected dask.distributed.Client, received"):
        DaskEngine(client="Client")

def test_submit_training_job(X_y_binary):
    class TestBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ["Logistic Regression Classifier"]
    pipeline = TestBinaryClassificationPipeline()

    engine = DaskEngine(client=client)

    engine.submit_training_job()