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