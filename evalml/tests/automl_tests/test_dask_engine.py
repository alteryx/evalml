import pytest
import time
from collections import namedtuple
from dask.distributed import Client
from sklearn import datasets

from evalml.objectives.utils import get_objective
from evalml.pipelines import BinaryClassificationPipeline
from evalml.automl.engine.dask_engine import DaskEngine
from evalml.automl.engine.sequential_engine import SequentialEngine

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
    automl_data = test_automl_data()

    class TestBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ["Logistic Regression Classifier"]

    pipeline = TestBinaryClassificationPipeline({})

    # Client needs to be instantiated inside the test or it hangs.
    client = Client()
    engine = DaskEngine(client=client)

    future = engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline)

    ans = future.get_result()
    print(ans)
    client.close()


def test_submit_multiple_training_jobs(X_y_binary):
    X, y = datasets.make_classification(n_samples=40000, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    automl_data = test_automl_data()

    class TestBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ["Logistic Regression Classifier"]

    num_pipelines = 8
    pipelines = []
    for i in range(num_pipelines):
        pipelines.append(TestBinaryClassificationPipeline({}))

    # Client needs to be instantiated inside the test or it hangs.

    parallel_time_start = time.time()
    client = Client()
    engine = DaskEngine(client=client)
    futures = []
    for pipeline in pipelines:
        futures.append(engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline))
    results = [f.get_result() for f in futures]
    parallel_time_taken = time.time()-parallel_time_start
    client.close()
    print(f"Parallel_time_taken: {parallel_time_taken}")
    print(results)

    sequential_time_start = time.time()
    engine = SequentialEngine()
    futures = []
    for pipeline in pipelines:
        futures.append(engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline))
    results = [f.get_result() for f in futures]
    sequential_time_taken = time.time()-sequential_time_start
    print(f"Sequential_time_taken: {sequential_time_taken}")
    print(results)

    print(sequential_time_taken, parallel_time_taken)
    import pdb; pdb.set_trace()

