import pytest

from evalml.automl.engines import EngineResult


def test_add_result(linear_regression_pipeline_class):
    single_pipeline = linear_regression_pipeline_class(parameters={'Linear Regressor': {'n_jobs': 1}})
    single_result = {
        'cv_data': [
            {'all_objective_scores': {}, 'binary_classificatio..._threshold': 0.5, 'score': 1},
            {'all_objective_scores': {}, 'binary_classificatio..._threshold': 0.5, 'score': 0},
        ],
        'training_time': 1.0,
        'cv_scores': [1, 0],
        'cv_score_mean': 0.5
    }    
    result = EngineResult()
    result.add_result(single_pipeline, single_result)
    assert result.early_stop == False
    assert len(result.completed_pipelines) == 1
    assert len(result.pipeline_results) == 1

    expected_error = "`completed_pipelines` must be PipelineBase or list(PipelineBase). Recieved <class 'bool'>."
    with pytest.raises(ValueError, match=expected_error):
        result.add_result(True, [])

    expected_error = "pipeline_results must be dict or list(dict). Recieved <class 'bool'>."
    with pytest.raises(ValueError, match=expected_error):
        result.add_result(single_pipeline, bool)

    processed_batch = [linear_regression_pipeline_class(parameters={'Linear Regressor': {'n_jobs': 1}}), linear_regression_pipeline_class(parameters={'Linear Regressor': {'n_jobs': 1}})]
    batch_result = [{
        'cv_data': [
            {'all_objective_scores': {}, 'binary_classificatio..._threshold': 0.5, 'score': 1},
            {'all_objective_scores': {}, 'binary_classificatio..._threshold': 0.5, 'score': 0},
        ],
        'training_time': 1.0,
        'cv_scores': [1, 0],
        'cv_score_mean': 0.5
    }, {
        'cv_data': [
            {'all_objective_scores': {}, 'binary_classificatio..._threshold': 0.5, 'score': 1},
            {'all_objective_scores': {}, 'binary_classificatio..._threshold': 0.5, 'score': 0},
        ],
        'training_time': 1.0,
        'cv_scores': [1, 0],
        'cv_score_mean': 0.5
    }]
    result = EngineResult()
    result.add_result(processed_batch, batch_result)
    assert result.early_stop == False
    assert len(result.completed_pipelines) == 2
    assert len(result.pipeline_results) == 2

    expected_error = "completed_pipelines must be PipelineBase or list(PipelineBase). Recieved <class 'list'>."
    with pytest.raises(ValueError, match=expected_error):
        result.add_result([True], [])

    expected_error = "pipeline_results must be dict or list(dict). Recieved <class 'bool'>."
    with pytest.raises(ValueError, match=expected_error):
        result.add_result(processed_batch, bool)
