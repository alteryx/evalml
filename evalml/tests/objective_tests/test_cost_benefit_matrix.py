import numpy as np
import pandas as pd
import pytest

from evalml import AutoMLSearch
from evalml.objectives import CostBenefitMatrix


def test_cbm_init():
    with pytest.raises(
        ValueError, match="Parameters to CostBenefitMatrix must all be numeric values."
    ):
        CostBenefitMatrix(
            true_positive=None, true_negative=-1, false_positive=-7, false_negative=-2
        )
    with pytest.raises(
        ValueError, match="Parameters to CostBenefitMatrix must all be numeric values."
    ):
        CostBenefitMatrix(
            true_positive=1, true_negative=-1, false_positive=None, false_negative=-2
        )
    with pytest.raises(
        ValueError, match="Parameters to CostBenefitMatrix must all be numeric values."
    ):
        CostBenefitMatrix(
            true_positive=1, true_negative=None, false_positive=-7, false_negative=-2
        )
    with pytest.raises(
        ValueError, match="Parameters to CostBenefitMatrix must all be numeric values."
    ):
        CostBenefitMatrix(
            true_positive=3, true_negative=-1, false_positive=-7, false_negative=None
        )


@pytest.mark.parametrize("optimize_thresholds", [True, False])
def test_cbm_objective_automl(optimize_thresholds, X_y_binary):
    X, y = X_y_binary
    cbm = CostBenefitMatrix(
        true_positive=10, true_negative=-1, false_positive=-7, false_negative=-2
    )
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        objective=cbm,
        max_iterations=2,
        optimize_thresholds=optimize_thresholds,
    )
    automl.search()

    pipeline = automl.best_pipeline
    pipeline.fit(X, y)
    predictions = pipeline.predict(X, cbm)
    assert not np.isnan(predictions).values.any()
    assert not np.isnan(pipeline.predict_proba(X)).values.any()
    assert not np.isnan(pipeline.score(X, y, [cbm])["Cost Benefit Matrix"])


@pytest.mark.parametrize("data_type", ["ww", "pd"])
def test_cbm_objective_function(data_type, make_data_type):
    y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    y_predicted = pd.Series([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])
    y_true = make_data_type(data_type, y_true)
    y_predicted = make_data_type(data_type, y_predicted)
    cbm = CostBenefitMatrix(
        true_positive=10, true_negative=-1, false_positive=-7, false_negative=-2
    )
    assert np.isclose(
        cbm.objective_function(y_true, y_predicted),
        ((3 * 10) + (-1 * 2) + (1 * -7) + (4 * -2)) / 10,
    )


def test_cbm_objective_function_floats():
    y_true = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    y_predicted = pd.Series([0, 0, 1, 0, 0, 0, 0, 1, 1, 1])
    cbm = CostBenefitMatrix(
        true_positive=5.1, true_negative=-1.2, false_positive=-6.7, false_negative=-0.1
    )
    assert np.isclose(
        cbm.objective_function(y_true, y_predicted),
        ((3 * 5.1) + (-1.2 * 2) + (1 * -6.7) + (4 * -0.1)) / 10,
    )


def test_cbm_input_contains_nan(X_y_binary):
    y_predicted = pd.Series([np.nan, 0, 0])
    y_true = pd.Series([1, 2, 1])
    cbm = CostBenefitMatrix(
        true_positive=10, true_negative=-1, false_positive=-7, false_negative=-2
    )
    with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
        cbm.score(y_true, y_predicted)

    y_true = pd.Series([np.nan, 0, 0])
    y_predicted = pd.Series([1, 2, 0])
    with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
        cbm.score(y_true, y_predicted)


def test_cbm_input_contains_inf(capsys):
    cbm = CostBenefitMatrix(
        true_positive=10, true_negative=-1, false_positive=-7, false_negative=-2
    )
    y_predicted = np.array([np.inf, 0, 0])
    y_true = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
        cbm.score(y_true, y_predicted)

    y_true = pd.Series([np.inf, 0, 0])
    y_predicted = pd.Series([1, 0, 0])
    with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
        cbm.score(y_true, y_predicted)


def test_cbm_different_input_lengths():
    cbm = CostBenefitMatrix(
        true_positive=10, true_negative=-1, false_positive=-7, false_negative=-2
    )
    y_predicted = pd.Series([0, 0])
    y_true = pd.Series([1])
    with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
        cbm.score(y_true, y_predicted)

    y_true = pd.Series([0, 0])
    y_predicted = pd.Series([1, 2, 0])
    with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
        cbm.score(y_true, y_predicted)


def test_cbm_zero_input_lengths():
    cbm = CostBenefitMatrix(
        true_positive=10, true_negative=-1, false_positive=-7, false_negative=-2
    )
    y_predicted = pd.Series([])
    y_true = pd.Series([])
    with pytest.raises(ValueError, match="Length of inputs is 0"):
        cbm.score(y_true, y_predicted)


def test_cbm_binary_more_than_two_unique_values():
    cbm = CostBenefitMatrix(
        true_positive=10, true_negative=-1, false_positive=-7, false_negative=-2
    )
    y_predicted = pd.Series([0, 1, 2])
    y_true = pd.Series([1, 0, 1])
    with pytest.raises(
        ValueError, match="y_predicted contains more than two unique values"
    ):
        cbm.score(y_true, y_predicted)

    y_true = pd.Series([0, 1, 2])
    y_predicted = pd.Series([1, 0, 1])
    with pytest.raises(ValueError, match="y_true contains more than two unique values"):
        cbm.score(y_true, y_predicted)
