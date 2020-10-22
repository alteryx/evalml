from unittest.mock import patch

from evalml.automl import AutoMLSearch
from evalml.automl.engines import SequentialEngine


class DummyAlgorithm:
    def __init__(self):
        self.batch_number = 0


@patch('evalml.pipelines.RegressionPipeline.score')
def test_load_new_engine(mock_score, X_y_binary, caplog):
    X, y = X_y_binary
    seq_engine = SequentialEngine()
    seq_engine.name = "Test Sequential Engine"
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_iterations=1)
    mock_score.return_value = {automl.objective.name: 1.234}
    automl.search(engine=seq_engine)
    out = caplog.text
    assert "Using Test Sequential Engine to process pipelines." in out


@patch('evalml.automl.engines.EngineBase._compute_cv_scores')
def test_evaluate_batch(mock_cv, X_y_binary, linear_regression_pipeline_class):
    X, y = X_y_binary
    seq_engine = SequentialEngine()
    seq_engine.load_data(X, y)
    automl = AutoMLSearch(X_train=X, y_train=y, problem_type='binary', max_time=1, max_iterations=1)
    automl._start = 0
    automl._automl_algorithm = DummyAlgorithm()
    automl._automl_algorithm.batch_number = 0
    seq_engine.load_search(automl)
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
    engine_result = seq_engine.evaluate_batch(pipeline_batch)
    fitted_pipelines = engine_result.completed_pipelines
    evaluation_results = engine_result.pipeline_results
    unprocessed_pipelines = engine_result.unprocessed_pipelines
    assert len(evaluation_results) == 3
    assert len(fitted_pipelines) == 3
    assert len(unprocessed_pipelines) == 0
