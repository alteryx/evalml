import time

from evalml.automl import Progress
from evalml.automl.automl_algorithm import DefaultAlgorithm


def test_progress_init():
    p = Progress()
    assert p.automl_algorithm is None
    assert p.max_batches is None
    assert p.max_iterations is None
    assert p.max_time is None
    assert p.tolerance is None
    assert p.patience is None

    p = Progress(
        automl_algorithm=DefaultAlgorithm,
        max_batches=3,
        max_iterations=2,
        max_time=1000,
    )

    assert p.automl_algorithm == DefaultAlgorithm
    assert p.max_batches == 3
    assert p.current_batches == 0
    assert p.max_iterations == 2
    assert p.current_iterations == 0
    assert p.max_time == 1000
    assert p.current_time is None


def test_progress_return_progress(X_y_binary, logistic_regression_binary_pipeline):
    X, y = X_y_binary
    algo = DefaultAlgorithm(X, y, problem_type="binary", sampler_name=None)
    p = Progress(
        automl_algorithm=algo,
        max_batches=3,
        max_time=10000,
    )
    p._start_time = time.time()
    mock_results = {"search_order": [0, 1, 2, 3], "pipeline_results": {}}
    scores = [
        0.84,
        0.95,
        0.84,
        0.96,
    ]  # 0.96 is only 1% greater so it doesn't trigger patience due to tolerance
    for id in mock_results["search_order"]:
        mock_results["pipeline_results"][id] = {}
        mock_results["pipeline_results"][id]["mean_cv_score"] = scores[id]
        mock_results["pipeline_results"][id][
            "pipeline_class"
        ] = logistic_regression_binary_pipeline.__class__

    assert p.should_continue(mock_results)

    progress_dict = p.return_progress()
    for progress in progress_dict:
        if progress["stopping_criteria"] == "max_time":
            assert progress["current_state"] >= 0
            assert progress["end_state"] == 10000
        elif progress["stopping_criteria"] == "max_iterations":
            assert progress["current_state"] == 4
            assert progress["end_state"] == sum(
                [algo.num_pipelines_per_batch(n) for n in range(3)],
            )
        elif progress["stopping_criteria"] == "max_batches":
            assert progress["current_state"] == algo.batch_number
            assert progress["end_state"] == 3
