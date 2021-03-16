import pytest
import time
from collections import namedtuple
from dask.distributed import Client
from sklearn import datasets
from unittest.mock import MagicMock, patch

import woodwork as ww
from evalml.objectives.utils import get_objective
from evalml.pipelines import BinaryClassificationPipeline
from evalml.automl.engine.engine_base import evaluate_pipeline
from evalml.automl.engine.dask_engine import DaskEngine
from evalml.automl.engine.sequential_engine import SequentialEngine
from evalml.preprocessing.data_splitters import RandomUnderSamplerCVSplit

AutoMLSearchStruct = namedtuple("AutoML",
                                "data_splitter problem_type objective additional_objectives optimize_thresholds error_callback random_seed ensembling_indices")
data_splitter = "K Fold"
problem_type = "binary"
objective = get_objective("Log Loss Binary")
additional_objectives = []
optimize_thresholds = False
error_callback = lambda x: 1
random_seed = 0
ensembling_indices = [0]
automl_data = AutoMLSearchStruct(data_splitter, problem_type, objective, additional_objectives,
                                 optimize_thresholds, error_callback, random_seed, ensembling_indices)


class TestBinaryClassificationPipeline(BinaryClassificationPipeline):
    component_graph = ["Logistic Regression Classifier"]

def mock_automl_data():
    data_splitter = MagicMock()
    problem_type = "binary"
    objective = get_objective("Log Loss Binary")
    additional_objectives = []
    optimize_thresholds = False
    error_callback = lambda x: 1
    random_seed = 0
    ensembling_indices = [0]
    automl_data = AutoMLSearchStruct(data_splitter, problem_type, objective, additional_objectives,
                              optimize_thresholds, error_callback, random_seed, ensembling_indices)
    return automl_data

def test_init():
    client = Client()
    engine = DaskEngine(client=client)
    assert engine.client == client

    with pytest.raises(TypeError, match="Expected dask.distributed.Client, received"):
        DaskEngine(client="Client")


def test_submit_training_job_single(X_y_binary):
    X,y = X_y_binary
    automl_data = mock_automl_data()

    class TestBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ["Logistic Regression Classifier"]
    pipeline = TestBinaryClassificationPipeline({})

    # Client needs to be instantiated inside the test or it hangs.
    client = Client()
    engine = DaskEngine(client=client)
    pipeline_future = engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline)
    pipeline_fitted = pipeline_future.get_result()
    assert pipeline_fitted._is_fitted
    client.close()


def test_submit_training_jobs_multiple(X_y_binary):
    X, y = datasets.make_classification(n_samples=40000, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    automl_data = mock_automl_data()

    class TestBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ["Logistic Regression Classifier"]

    def fit_pipelines(engine):
        num_pipelines = 4
        pipelines = []
        for i in range(num_pipelines):
            pipelines.append(TestBinaryClassificationPipeline({}))

        time_start = time.time()
        futures = []
        for pipeline in pipelines:
            futures.append(engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline))
        results = [f.get_result() for f in futures]
        time_taken = time.time()-time_start
        return results, time_taken

    # Client needs to be instantiated inside the test or it hangs.
    client = Client()
    par_pipelines, par_time_taken = fit_pipelines(DaskEngine(client=client))
    for pipeline in par_pipelines:
        assert pipeline._is_fitted
    client.close()

    seq_pipelines, seq_time_taken = fit_pipelines(SequentialEngine())
    for pipeline in seq_pipelines:
        assert pipeline._is_fitted

    print(seq_time_taken, par_time_taken)

@patch(evaluate_pipeline)
def test_submit_evaluate_job_single(mock_evaluate_pipeline, X_y_binary):
    mock_evaluate_pipeline.return_value = 1

    X,y = X_y_binary
    # automl_data = mock_automl_data()

    pipeline = TestBinaryClassificationPipeline({})

    # Client needs to be instantiated inside the test or it hangs.
    client = Client()
    engine = DaskEngine(client=client)
    pipeline_future = engine.submit_evaluation_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                   automl_data=automl_data, pipeline=pipeline)
    pipeline_fitted = pipeline_future.get_result()

    client.close()
    import pdb; pdb.set_trace()

