import pytest

from evalml.automl import AutoMLSearch
from evalml.pipelines.components.transformers import ReplaceNullableTypes
from evalml.problem_types import ProblemTypes, is_time_series


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("automl_algorithm", ["iterative", "default"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize(
    "test_description, column_names",
    [
        (
            "all null",
            ["dates", "all_null"],
        ),  # Should result only in Drop Null Columns Transformer
        ("only null int", ["int_null"]),
        ("only null bool", ["bool_null"]),
        ("only null age", ["age_null"]),
        ("nullable types", ["numerical", "int_null", "bool_null", "age_null"]),
        ("just nullable target", ["dates", "numerical"]),
    ],
)
def test_nullable_types_builds_pipelines(
    automl_algorithm,
    problem_type,
    input_type,
    test_description,
    column_names,
    get_test_data_from_configuration,
):
    parameters = {}
    if is_time_series(problem_type):
        parameters = {
            "time_index": "dates",
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 3,
        }

    X, y = get_test_data_from_configuration(
        input_type,
        problem_type,
        column_names=column_names,
        nullable_target=True if "nullable target" in test_description else False,
    )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        problem_configuration=parameters,
        automl_algorithm=automl_algorithm,
    )
    if automl_algorithm == "iterative":
        pipelines = [pl.name for pl in automl.allowed_pipelines]
    elif automl_algorithm == "default":
        # TODO: Upon resolution of GH Issue #3186, increase the num of batches.
        for _ in range(2):
            pipelines = [pl.name for pl in automl.automl_algorithm.next_batch()]

    # A check to make sure we actually retrieve constructed pipelines from the algo.
    assert len(pipelines) > 0

    if test_description in [
        "only null int",
        "only null bool",
        "only null age",
        "nullable types",
        "just nullable target",
    ]:
        if input_type == "pd":
            assert not any([ReplaceNullableTypes.name in pl for pl in pipelines])
        elif input_type == "ww":
            assert all([ReplaceNullableTypes.name in pl for pl in pipelines])
    else:
        assert not any([ReplaceNullableTypes.name in pl for pl in pipelines])
