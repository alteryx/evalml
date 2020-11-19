import pandas as pd

from evalml.data_checks import (
    DataCheckError,
    DataCheckResults,
    DataCheckWarning
)


def test_data_check_results_equality():
    results_error = DataCheckResults(errors=[DataCheckError("error one", "error name")])
    results_warnings = DataCheckResults(warnings=[DataCheckWarning("warning one", "warning name")])
    results_errors_and_warnings = DataCheckResults(errors=[DataCheckError("error one", "error name")],
                                                   warnings=[DataCheckWarning("warning one", "warning name")])
    assert results_error == results_error
    assert results_warnings == results_warnings
    assert results_errors_and_warnings == results_errors_and_warnings
    assert results_error == DataCheckResults(errors=[DataCheckError("error one", "error name")])
    assert results_warnings == DataCheckResults(warnings=[DataCheckWarning("warning one", "warning name")])
    assert results_errors_and_warnings == DataCheckResults(errors=[DataCheckError("error one", "error name")],
                                                           warnings=[DataCheckWarning("warning one", "warning name")])
    assert results_error != results_warnings
    assert results_error != results_errors_and_warnings
    assert results_warnings != results_error
    assert results_warnings != results_errors_and_warnings
    assert results_errors_and_warnings != results_error
    assert results_errors_and_warnings != results_warnings
