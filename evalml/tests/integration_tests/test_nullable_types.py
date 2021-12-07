import pytest

from evalml.automl import AutoMLSearch
from evalml.pipelines.components.transformers import ReplaceNullableTypes
from evalml.problem_types import ProblemTypes, is_time_series


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize(
    "test_description, column_names",
    [
        ("all_null", ["dates", "all_null"]),
        ("nullable_types", ["dates", "numerical", "int_null", "bool_null"]),
    ],
)
def test_nullable_types_builds_pipelines(
    problem_type,
    input_type,
    test_description,
    column_names,
    get_test_data_from_configuration,
):
    parameters = {}
    if is_time_series(problem_type):
        parameters = {
            "date_index": "dates",
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 3,
        }

    X, y = get_test_data_from_configuration(
        input_type,
        problem_type,
        column_names=column_names,
    )

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type=problem_type,
        problem_configuration=parameters,
    )
    pipelines = [pl.name for pl in automl.allowed_pipelines]

    if test_description == "nullable_types":
        if input_type == "pd":
            assert not any([ReplaceNullableTypes.name in pl for pl in pipelines])
        elif input_type == "ww":
            assert all([ReplaceNullableTypes.name in pl for pl in pipelines])
    else:
        assert not any([ReplaceNullableTypes.name in pl for pl in pipelines])
