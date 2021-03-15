import pytest
from collections import namedtuple
from dask.distributed import Client

from evalml.objectives.utils import get_objective
from evalml.pipelines import BinaryClassificationPipeline
from evalml.automl.engine.dask_engine import DaskEngine

# client = Client()

AutoMLSearchStruct = namedtuple("AutoML",
                                "data_splitter problem_type objective additional_objectives optimize_thresholds error_callback random_seed")


def test_automl_data():
    data_splitter = "Stratified K Fold"
    problem_type = "binary"
    objective = get_objective("Log Loss Binary")
    additional_objectives = []
    optimize_thresholds = False
    error_callback = lambda x: 1
    random_seed = 0
    automl_data = AutoMLSearchStruct(data_splitter, problem_type, objective, additional_objectives,
                              optimize_thresholds, error_callback, random_seed)
    return automl_data

def test_init():
    client = Client()
    engine = DaskEngine(client=client)
    assert engine.client == client

    with pytest.raises(TypeError, match="Expected dask.distributed.Client, received"):
        DaskEngine(client="Client")


def test_submit_single_training_job(X_y_binary):
    X,y = X_y_binary

    class TestBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ["Logistic Regression Classifier"]

    pipeline = TestBinaryClassificationPipeline({})

    automl_data = test_automl_data()
    # Client needs to be instantiated inside the test or it hangs.
    client = Client()
    engine = DaskEngine(client=client)

    res = engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline)

    # ans = client.gather(res.work)
    ans = res.get_result()
    print(ans)
    client.close()
    import pdb; pdb.set_trace()


def test_submit_multiple_training_jobs(X_y_binary):
    X,y = X_y_binary

    class TestBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ["Logistic Regression Classifier"]

    pipelines = [TestBinaryClassificationPipeline({}),
                 TestBinaryClassificationPipeline({}),
                 TestBinaryClassificationPipeline({}),
                 TestBinaryClassificationPipeline({})]

    automl_data = test_automl_data()
    # Client needs to be instantiated inside the test or it hangs.
    client = Client()
    engine = DaskEngine(client=client)

    ress = []
    for pipeline in pipelines:
        ress.append(engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline))

    ans = [f.get_result() for f in ress]
    print(ans)
    client.close()
    import pdb; pdb.set_trace()
