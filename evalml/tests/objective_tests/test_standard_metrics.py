from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import matthews_corrcoef as sk_matthews_corrcoef

from evalml.objectives import (
    F1,
    MAPE,
    MSE,
    AccuracyBinary,
    AccuracyMulticlass,
    BalancedAccuracyBinary,
    BalancedAccuracyMulticlass,
    BinaryClassificationObjective,
    CostBenefitMatrix,
    ExpVariance,
    F1Macro,
    F1Micro,
    F1Weighted,
    LogLossBinary,
    MCCBinary,
    MCCMulticlass,
    MeanSquaredLogError,
    Precision,
    PrecisionMacro,
    PrecisionMicro,
    PrecisionWeighted,
    Recall,
    RecallMacro,
    RecallMicro,
    RecallWeighted,
    RootMeanSquaredError,
    RootMeanSquaredLogError,
)
from evalml.objectives.utils import (
    _all_objectives_dict,
    get_non_core_objectives,
)

EPS = 1e-5
all_automl_objectives = _all_objectives_dict()
all_automl_objectives = {
    name: class_()
    for name, class_ in all_automl_objectives.items()
    if class_ not in get_non_core_objectives()
}


def test_input_contains_nan():
    y_predicted = np.array([np.nan, 0, 0])
    y_true = np.array([1, 2, 1])
    for objective in all_automl_objectives.values():
        with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
            objective.score(y_true, y_predicted)

    y_true = np.array([np.nan, 0, 0])
    y_predicted = np.array([1, 2, 0])
    for objective in all_automl_objectives.values():
        with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
            objective.score(y_true, y_predicted)

    y_true = np.array([1, 0])
    y_predicted_proba = np.array([[1, np.nan], [0.1, 0]])
    for objective in all_automl_objectives.values():
        if objective.score_needs_proba:
            with pytest.raises(
                ValueError, match="y_predicted contains NaN or infinity"
            ):
                objective.score(y_true, y_predicted_proba)


def test_input_contains_inf():
    y_predicted = np.array([np.inf, 0, 0])
    y_true = np.array([1, 0, 0])
    for objective in all_automl_objectives.values():
        with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
            objective.score(y_true, y_predicted)

    y_true = np.array([np.inf, 0, 0])
    y_predicted = np.array([1, 0, 0])
    for objective in all_automl_objectives.values():
        with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
            objective.score(y_true, y_predicted)

    y_true = np.array([1, 0])
    y_predicted_proba = np.array([[1, np.inf], [0.1, 0]])
    for objective in all_automl_objectives.values():
        if objective.score_needs_proba:
            with pytest.raises(
                ValueError, match="y_predicted contains NaN or infinity"
            ):
                objective.score(y_true, y_predicted_proba)


def test_different_input_lengths():
    y_predicted = np.array([0, 0])
    y_true = np.array([1])
    for objective in all_automl_objectives.values():
        with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
            objective.score(y_true, y_predicted)

    y_true = np.array([0, 0])
    y_predicted = np.array([1, 2, 0])
    for objective in all_automl_objectives.values():
        with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
            objective.score(y_true, y_predicted)


def test_zero_input_lengths():
    y_predicted = np.array([])
    y_true = np.array([])
    for objective in all_automl_objectives.values():
        with pytest.raises(ValueError, match="Length of inputs is 0"):
            objective.score(y_true, y_predicted)


def test_probabilities_not_in_0_1_range():
    y_predicted = np.array([0.3, 1.001, 0.3])
    y_true = np.array([1, 0, 1])
    for objective in all_automl_objectives.values():
        if objective.score_needs_proba:
            with pytest.raises(
                ValueError, match="y_predicted contains probability estimates"
            ):
                objective.score(y_true, y_predicted)

    y_predicted = np.array([0.3, -0.001, 0.3])
    y_true = np.array([1, 0, 1])
    for objective in all_automl_objectives.values():
        if objective.score_needs_proba:
            with pytest.raises(
                ValueError, match="y_predicted contains probability estimates"
            ):
                objective.score(y_true, y_predicted)

    y_true = np.array([1, 0])
    y_predicted_proba = np.array([[1, 3], [0.1, 0]])
    for objective in all_automl_objectives.values():
        if objective.score_needs_proba:
            with pytest.raises(
                ValueError, match="y_predicted contains probability estimates"
            ):
                objective.score(y_true, y_predicted_proba)


def test_negative_with_log():
    y_predicted = np.array([-1, 10, 30])
    y_true = np.array([-1, 0, 1])
    for objective in [MeanSquaredLogError(), RootMeanSquaredLogError()]:
        with pytest.raises(
            ValueError,
            match="Mean Squared Logarithmic Error cannot be used when targets contain negative values.",
        ):
            objective.score(y_true, y_predicted)


def test_binary_more_than_two_unique_values():
    y_predicted = np.array([0, 1, 2])
    y_true = np.array([1, 0, 1])
    for objective in all_automl_objectives.values():
        if (
            isinstance(objective, BinaryClassificationObjective)
            and not objective.score_needs_proba
        ):
            with pytest.raises(
                ValueError, match="y_predicted contains more than two unique values"
            ):
                objective.score(y_true, y_predicted)

    y_true = np.array([0, 1, 2])
    y_predicted = np.array([1, 0, 1])
    for objective in all_automl_objectives.values():
        if (
            isinstance(objective, BinaryClassificationObjective)
            and not objective.score_needs_proba
        ):
            with pytest.raises(
                ValueError, match="y_true contains more than two unique values"
            ):
                objective.score(y_true, y_predicted)


def test_accuracy_binary():
    obj = AccuracyBinary()
    assert obj.score(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == pytest.approx(
        0.0, EPS
    )
    assert obj.score(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == pytest.approx(
        0.5, EPS
    )
    assert obj.score(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == pytest.approx(
        1.0, EPS
    )
    assert (
        obj.score(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
            sample_weight=np.array([0.5, 0, 0, 0.5]),
        )
        == pytest.approx(1.0, EPS)
    )


def test_accuracy_multi():
    obj = AccuracyMulticlass()
    assert obj.score(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == pytest.approx(
        0.0, EPS
    )
    assert obj.score(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == pytest.approx(
        0.5, EPS
    )
    assert obj.score(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == pytest.approx(
        1.0, EPS
    )
    assert obj.score(
        np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 0, 0, 0, 0])
    ) == pytest.approx(1 / 3.0, EPS)
    assert obj.score(
        np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 1, 1, 2, 2])
    ) == pytest.approx(1 / 3.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 0, 1]),
            sample_weight=np.array([0.5, 0, 0, 0.5]),
        )
        == pytest.approx(1.0, EPS)
    )


def test_balanced_accuracy_binary():
    obj = BalancedAccuracyBinary()
    assert obj.score(
        np.array([0, 1, 0, 0, 1, 0]), np.array([0, 1, 0, 0, 0, 1])
    ) == pytest.approx(0.625, EPS)

    assert obj.score(
        np.array([0, 1, 0, 0, 1, 0]), np.array([0, 1, 0, 0, 1, 0])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 1, 0, 0, 1, 0]), np.array([1, 0, 1, 1, 0, 1])
    ) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([0, 1, 0, 0, 0, 1]),
            sample_weight=np.array([0.25, 0.25, 0.25, 0.25, 0, 0]),
        )
        == pytest.approx(1.0, EPS)
    )


def test_balanced_accuracy_multi():
    obj = BalancedAccuracyMulticlass()
    assert obj.score(
        np.array([0, 1, 2, 0, 1, 2, 3]), np.array([0, 0, 2, 0, 0, 2, 3])
    ) == pytest.approx(0.75, EPS)

    assert obj.score(
        np.array([0, 1, 2, 0, 1, 2, 3]), np.array([0, 1, 2, 0, 1, 2, 3])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 1, 2, 0, 1, 2, 3]), np.array([1, 0, 3, 1, 2, 1, 0])
    ) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 1, 2, 0, 1, 2, 3]),
            np.array([0, 0, 2, 0, 0, 2, 3]),
            sample_weight=np.array([0, 0.25, 0, 0, 0.25, 0, 0]),
        )
        == pytest.approx(0.0, EPS)
    )


def test_f1_binary():
    obj = F1()
    assert obj.score(
        np.array([0, 1, 0, 0, 1, 0]), np.array([0, 1, 0, 0, 0, 1])
    ) == pytest.approx(0.5, EPS)

    assert obj.score(
        np.array([0, 1, 0, 0, 1, 1]), np.array([0, 1, 0, 0, 1, 1])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 0, 1, 0]), np.array([0, 1, 0, 0, 0, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]), np.array([0, 0])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([0, 1, 0, 0, 0, 1]),
            sample_weight=np.array([0, 0, 0, 0, 1, 0]),
        )
        == pytest.approx(0.0, EPS)
    )


def test_f1_micro_multi():
    obj = F1Micro()
    assert obj.score(
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([2, 2, 2, 0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([1, 2]), np.array([0, 0])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            sample_weight=np.array([1, 0, 0, 1, 0, 0, 0, 0, 1]),
        )
        == pytest.approx(1 / 3.0, EPS)
    )


def test_f1_macro_multi():
    obj = F1Macro()
    assert obj.score(
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(2 * (1 / 3.0) * (1 / 9.0) / (1 / 3.0 + 1 / 9.0), EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([2, 2, 2, 0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([1, 2]), np.array([0, 0])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([1, 0, 0, 1, 1, 1, 2, 2, 2]),
            sample_weight=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
        )
        == pytest.approx(0.0, EPS)
    )


def test_f1_weighted_multi():
    obj = F1Weighted()
    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    ) == pytest.approx(2 * (1 / 3.0) * (1 / 9.0) / (1 / 3.0 + 1 / 9.0), EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]), np.array([1, 2])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            sample_weight=np.array([1, 0, 1, 0, 0, 0, 0, 0, 0]),
        )
        == pytest.approx(1.0, EPS)
    )


def test_precision_binary():
    obj = Precision()
    assert obj.score(
        np.array([1, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1]), np.array([1, 1, 1, 1, 1, 1])
    ) == pytest.approx(0.5, EPS)

    assert obj.score(
        np.array([0, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0])
    ) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 1, 1, 1]),
            sample_weight=np.array([0, 100, 0, 0, 0, 0]),
        )
        == pytest.approx(0.0, EPS)
    )


def test_precision_micro_multi():
    obj = PrecisionMicro()
    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    ) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]), np.array([1, 2])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            sample_weight=np.array([100, 0, 0, 0, 0, 0, 0, 0, 0]),
        )
        == pytest.approx(1, EPS)
    )


def test_precision_macro_multi():
    obj = PrecisionMacro()
    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    ) == pytest.approx(1 / 9.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]), np.array([1, 2])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            sample_weight=np.array([0, 0, 0, 0, 0, 0, 0, 5, 5]),
        )
        == pytest.approx(0.0, EPS)
    )


def test_precision_weighted_multi():
    obj = PrecisionWeighted()
    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    ) == pytest.approx(1 / 9.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]), np.array([1, 2])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            sample_weight=np.array([5, 0, 0, 0, 0, 0, 0, 0, 5]),
        )
        == pytest.approx(1 / 4.0, EPS)
    )


def test_recall_binary():
    obj = Recall()
    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1]), np.array([1, 1, 1, 1, 1, 1])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 0, 0, 0])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(
        np.array([1, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1])
    ) == pytest.approx(0.5, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1]),
            np.array([1, 1, 1, 1, 1, 1]),
            sample_weight=np.array([1, 0, 1, 0, 0, 1]),
        )
        == pytest.approx(1.0, EPS)
    )


def test_recall_micro_multi():
    obj = RecallMicro()
    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    ) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]), np.array([1, 2])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            sample_weight=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )
        == pytest.approx(0.0, EPS)
    )


def test_recall_macro_multi():
    obj = RecallMacro()
    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    ) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]), np.array([1, 2])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            sample_weight=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
        )
        == pytest.approx(0.0, EPS)
    )


def test_recall_weighted_multi():
    obj = RecallWeighted()
    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    ) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ) == pytest.approx(1.0, EPS)

    assert obj.score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
    ) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]), np.array([1, 2])) == pytest.approx(0.0, EPS)

    assert (
        obj.score(
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            sample_weight=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
        )
        == pytest.approx(0.0, EPS)
    )


def test_log_linear_model():
    obj = MeanSquaredLogError()
    root_obj = RootMeanSquaredLogError()

    sample_weight = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

    s1_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s1_actual = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    s2_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s2_actual = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    s3_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s3_actual = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])

    assert obj.score(s1_predicted, s1_actual) == pytest.approx(0.562467324910)
    assert obj.score(s2_predicted, s2_actual) == pytest.approx(0)
    assert obj.score(s3_predicted, s3_actual) == pytest.approx(0.617267976207983)

    assert obj.score(
        s1_predicted, s1_actual, sample_weight=sample_weight
    ) == pytest.approx(0.48045301391820133)

    assert root_obj.score(s1_predicted, s1_actual) == pytest.approx(
        np.sqrt(0.562467324910)
    )
    assert root_obj.score(s2_predicted, s2_actual) == pytest.approx(0)
    assert root_obj.score(s3_predicted, s3_actual) == pytest.approx(
        np.sqrt(0.617267976207983)
    )

    assert root_obj.score(
        s3_predicted, s3_actual, sample_weight=sample_weight
    ) == pytest.approx(0.6931471805599453)


def test_mse_linear_model():
    obj = MSE()
    root_obj = RootMeanSquaredError()

    sample_weight = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

    s1_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s1_actual = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    s2_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s2_actual = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    s3_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s3_actual = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])

    assert obj.score(s1_predicted, s1_actual) == pytest.approx(5.0 / 3.0)
    assert obj.score(s2_predicted, s2_actual) == pytest.approx(0)
    assert obj.score(s3_predicted, s3_actual) == pytest.approx(2.0)

    assert obj.score(
        s1_predicted, s1_actual, sample_weight=sample_weight
    ) == pytest.approx(1.0)

    assert root_obj.score(s1_predicted, s1_actual) == pytest.approx(np.sqrt(5.0 / 3.0))
    assert root_obj.score(s2_predicted, s2_actual) == pytest.approx(0)
    assert root_obj.score(s3_predicted, s3_actual) == pytest.approx(np.sqrt(2.0))

    assert root_obj.score(
        s3_predicted, s3_actual, sample_weight=sample_weight
    ) == pytest.approx(1.0)


def test_mcc_catches_warnings():
    y_true = [1, 0, 1, 1]
    y_predicted = [0, 0, 0, 0]
    with pytest.warns(RuntimeWarning) as record:
        sk_matthews_corrcoef(y_true, y_predicted)
        assert "invalid value" in str(record[-1].message)
    with pytest.warns(None) as record:
        MCCBinary().objective_function(y_true, y_predicted)
        MCCMulticlass().objective_function(y_true, y_predicted)
        assert len(record) == 0


def test_mape_time_series_model():
    obj = MAPE()

    s1_actual = np.array([0, 0, 1, 1, 1, 1, 2, 0, 2])
    s1_predicted = np.array([0, 1, 0, 1, 1, 2, 1, 2, 0])

    s2_actual = np.array([-1, -2, 1, 3])
    s2_predicted = np.array([1, 2, -1, -3])

    s3_actual = np.array([1, 2, 4, 2, 1, 2])
    s3_predicted = np.array([0, 2, 2, 1, 3, 2])

    with pytest.raises(
        ValueError,
        match="Mean Absolute Percentage Error cannot be used when targets contain the value 0.",
    ):
        obj.score(s1_actual, s1_predicted)
    assert obj.score(s2_actual, s2_predicted) == pytest.approx(8 / 4 * 100)
    assert obj.score(s3_actual, s3_predicted) == pytest.approx(4 / 6 * 100)
    assert obj.score(
        pd.Series(s3_actual, index=range(-12, -6)), s3_predicted
    ) == pytest.approx(4 / 6 * 100)
    assert (
        obj.score(
            pd.Series(s2_actual, index=range(10, 14)),
            pd.Series(s2_predicted, index=range(20, 24)),
        )
        == pytest.approx(8 / 4 * 100)
    )


@pytest.mark.parametrize("objective_class", _all_objectives_dict().values())
def test_calculate_percent_difference(objective_class):
    score = 5
    reference_score = 10
    denominator = 1 if objective_class.is_bounded_like_percentage else reference_score

    change = (
        (-1) ** (not objective_class.greater_is_better) * (score - reference_score)
    ) / denominator
    answer = 100 * change

    assert (
        objective_class.calculate_percent_difference(score, reference_score) == answer
    )
    assert objective_class.perfect_score is not None


@pytest.mark.parametrize(
    "objective_class,nan_value",
    product(_all_objectives_dict().values(), [None, np.nan]),
)
def test_calculate_percent_difference_with_nan(objective_class, nan_value):

    assert pd.isna(objective_class.calculate_percent_difference(nan_value, 2))
    assert pd.isna(objective_class.calculate_percent_difference(-1, nan_value))
    assert pd.isna(objective_class.calculate_percent_difference(nan_value, nan_value))


@pytest.mark.parametrize("baseline_score", [0, 1e-11])
@pytest.mark.parametrize("objective_class", _all_objectives_dict().values())
def test_calculate_percent_difference_when_baseline_0_or_close_to_0(
    objective_class, baseline_score
):
    percent_difference = objective_class.calculate_percent_difference(2, baseline_score)
    if objective_class.is_bounded_like_percentage:
        assert (
            percent_difference
            == ((-1) ** (not objective_class.greater_is_better))
            * (2 - baseline_score)
            * 100
        )
    else:
        assert np.isinf(percent_difference)


def test_calculate_percent_difference_negative_and_equal_numbers():

    assert (
        CostBenefitMatrix.calculate_percent_difference(score=5, baseline_score=5) == 0
    )
    assert (
        CostBenefitMatrix.calculate_percent_difference(
            score=5.003, baseline_score=5.003 - 1e-11
        )
        == 0
    )
    assert (
        CostBenefitMatrix.calculate_percent_difference(score=-5, baseline_score=-10)
        == 50
    )
    assert (
        CostBenefitMatrix.calculate_percent_difference(score=-10, baseline_score=-5)
        == -100
    )
    assert (
        CostBenefitMatrix.calculate_percent_difference(score=-5, baseline_score=10)
        == -150
    )
    assert (
        CostBenefitMatrix.calculate_percent_difference(score=10, baseline_score=-5)
        == 300
    )

    # These values are not possible for LogLossBinary but we need them for 100% coverage
    # We might add an objective where lower is better that can take negative values in the future
    assert (
        LogLossBinary.calculate_percent_difference(score=-5, baseline_score=-10) == -50
    )
    assert (
        LogLossBinary.calculate_percent_difference(
            score=5.003, baseline_score=5.003 + 1e-11
        )
        == 0
    )
    assert (
        LogLossBinary.calculate_percent_difference(score=-10, baseline_score=-5) == 100
    )
    assert (
        LogLossBinary.calculate_percent_difference(score=-5, baseline_score=10) == 150
    )
    assert (
        LogLossBinary.calculate_percent_difference(score=10, baseline_score=-5) == -300
    )

    # Verify percent_difference is 0 when numbers are close to equal for objective that is bounded in [0, 1]
    assert AccuracyBinary.calculate_percent_difference(score=5, baseline_score=5) == 0
    assert (
        AccuracyBinary.calculate_percent_difference(
            score=5.003, baseline_score=5.003 + 1e-11
        )
        == 0
    )


def test_calculate_percent_difference_small():
    expected_value = 100 * -1 * np.abs(1e-9 / (1e-9))
    assert np.isclose(
        ExpVariance.calculate_percent_difference(score=0, baseline_score=1e-9),
        expected_value,
        atol=1e-8,
    )
    assert ExpVariance.calculate_percent_difference(score=0, baseline_score=1e-10) == 0
    assert (
        ExpVariance.calculate_percent_difference(score=2e-10, baseline_score=1e-10) == 0
    )
    assert np.isinf(
        ExpVariance.calculate_percent_difference(score=1e-9, baseline_score=0)
    )
    assert np.isinf(
        ExpVariance.calculate_percent_difference(score=0.1, baseline_score=1e-11)
    )

    expected_value = 100 * np.abs(1e-9)
    assert np.isclose(
        AccuracyBinary.calculate_percent_difference(score=0, baseline_score=1e-9),
        expected_value,
        atol=1e-6,
    )
    assert (
        AccuracyBinary.calculate_percent_difference(score=0, baseline_score=1e-10) == 0
    )
    assert (
        AccuracyBinary.calculate_percent_difference(score=2e-10, baseline_score=1e-10)
        == 0
    )
    assert np.isclose(
        AccuracyBinary.calculate_percent_difference(score=1e-9, baseline_score=0),
        expected_value,
        atol=1e-6,
    )
    assert np.isclose(
        AccuracyBinary.calculate_percent_difference(score=0.1, baseline_score=1e-11),
        100 * np.abs(0.1 - 1e-11),
        atol=1e-6,
    )
