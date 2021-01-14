from unittest.mock import patch

from distributed.utils_test import client
from distributed import Future

from evalml.automl import AutoMLSearch
from evalml.automl.engines import DaskEngine


class DummyAlgorithm:
    def __init__(self):
        self.batch_number = 0


@patch('evalml.pipelines.RegressionPipeline.score')
def test_load_new_engine(mock_score, X_y_binary, caplog):
    def test_client(client): 
        X, y = X_y_binary
        dask_engine = DaskEngine(client)
        dask_engine.name = "Test Dask Engine"
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_iterations=1)
        mock_score.return_value = {automl.objective.name: 1.234}
        automl.search(engine=dask_engine)
        out = caplog.text
        assert "Using Test Dask Engine to process pipelines." in out


def test_load_dataset(X_y_binary):
    def test_client(client):
        X, y = X_y_binary
        dask_engine = DaskEngine(client)
        dask_engine.load_data(X, y)
        assert isinstance(dask_engine.X_future, Future)
        assert isinstance(dask_engine.y_future, Future)
        assert dask_engine.X_future.key in client.futures
        assert dask_engine.y_future.key in client.futures


@patch('evalml.automl.engines.EngineBase._compute_cv_scores')
def test_evaluate_batch(mock_cv, X_y_binary, linear_regression_pipeline_class):
    def test_client(client):
        X, y = X_y_binary
        dask_engine = DaskEngine(client)
        dask_engine.load_data(X, y)
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_iterations=1)
        automl._start = 0
        automl._automl_algorithm = DummyAlgorithm()
        automl._automl_algorithm.batch_number = 0
        dask_engine.load_search(automl)
        mock_regression_pipeline = linear_regression_pipeline_class(parameters={'Linear Regressor': {'n_jobs': 1}})
        pipeline_batch = [mock_regression_pipeline, mock_regression_pipeline, mock_regression_pipeline]
        score_dict = {
            'cv_data': [
                {'all_objective_scores': {}, 'binary_classificatio..._threshold': 0.5, 'score': 1},
                {'all_objective_scores': {}, 'binary_classificatio..._threshold': 0.5, 'score': 0},
            ],
            'training_time': 1.0,
            'cv_scores': [1, 0],
            'cv_score_mean': 0.5
        }
        mock_cv.return_value = (mock_regression_pipeline, score_dict)
        engine_result = dask_engine.evaluate_batch(pipeline_batch)
        fitted_pipelines = engine_result.completed_pipelines
        evaluation_results = engine_result.pipeline_results
        unprocessed_pipelines = engine_result.unprocessed_pipelines
        assert len(evaluation_results) == 3
        assert len(fitted_pipelines) == 3
        assert len(unprocessed_pipelines) == 0
