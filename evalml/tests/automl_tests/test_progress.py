import time

import pytest

from evalml.automl import Progress
from evalml.automl.automl_algorithm import DefaultAlgorithm
from evalml.objectives import Gini


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
    assert p.current_batch == 0
    assert p.max_iterations == 2
    assert p.current_iterations == 0
    assert p.max_time == 1000
    assert p.current_time is None


@pytest.mark.parametrize(
    "max_time, max_batches, max_iterations, early_stopping",
    [
        (10000, 5, None, False),
        (10000, 5, None, True),
    ],
)
def test_progress_should_continue(
    max_time,
    max_batches,
    max_iterations,
    early_stopping,
    X_y_binary,
    logistic_regression_binary_pipeline,
):
    X, y = X_y_binary
    algo = DefaultAlgorithm(X, y, problem_type="binary", sampler_name=None)

    tolerance = 0.05 if early_stopping else None
    patience = 3 if early_stopping else None
    objective = Gini if early_stopping else None

    p = Progress(
        automl_algorithm=algo,
        max_batches=max_batches,
        max_time=max_time,
        max_iterations=max_iterations,
        tolerance=tolerance,
        patience=patience,
        objective=objective,
    )

    p.start_time = time.time()
    search_order = [0, 1, 2, 3]
    mock_results = {"search_order": [], "pipeline_results": {}}
    scores = [
        0.84,
        0.95,
        0.84,
        0.96,
    ]  # 0.96 is only 1% greater so it doesn't trigger patience due to tolerance
    for id in search_order:
        mock_results["search_order"].append(id)
        mock_results["pipeline_results"][id] = {}
        mock_results["pipeline_results"][id]["mean_cv_score"] = scores[id]
        mock_results["pipeline_results"][id][
            "pipeline_class"
        ] = logistic_regression_binary_pipeline.__class__
        assert p.should_continue(mock_results)

    assert p.should_continue(
        mock_results,
    )  # test that we don't trigger tolerance when results don't change

    mock_results["search_order"].append(4)
    mock_results["pipeline_results"][4] = {}
    mock_results["pipeline_results"][4]["mean_cv_score"] = 0.97
    mock_results["pipeline_results"][4][
        "pipeline_class"
    ] = logistic_regression_binary_pipeline.__class__

    if early_stopping:
        assert not p.should_continue(mock_results)
    else:
        assert p.should_continue(mock_results)


def test_progress_return_progress(X_y_binary, logistic_regression_binary_pipeline):
    X, y = X_y_binary
    algo = DefaultAlgorithm(X, y, problem_type="binary", sampler_name=None)
    p = Progress(
        automl_algorithm=algo,
        max_batches=3,
        max_time=10000,
    )
    p.start_time = time.time()
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

    p.should_continue(mock_results)
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
