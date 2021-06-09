from evalml.automl.automl_algorithm import EvalMLAlgorithm


def test_iterative_algorithm_init():
    algo = EvalMLAlgorithm()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []
