import numpy as np
import pytest

from evalml.objectives import (
    F1,
    AccuracyBinary,
    AccuracyMulticlass,
    BalancedAccuracyBinary,
    BalancedAccuracyMulticlass,
    BinaryClassificationObjective,
    F1Macro,
    F1Micro,
    F1Weighted,
    Precision,
    PrecisionMacro,
    PrecisionMicro,
    PrecisionWeighted,
    Recall,
    RecallMacro,
    RecallMicro,
    RecallWeighted
)
from evalml.objectives.utils import OPTIONS

EPS = 1e-5


def test_input_contains_nan(capsys):
    y_pred = np.array([np.nan, 0, 0])
    y_true = np.array([1, 2, 1])
    for objective in OPTIONS.values():
        out, err = capsys.readouterr()
        objective.score(y_true, y_pred)
        assert out == "WARNING: y_predicted contains NaN or infinity"
        assert err == ""
    y_true = np.array([np.nan, 0, 0])
    y_pred = np.array([1, 2, 0])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
            objective.score(y_true, y_pred)


def test_input_contains_inf(capsys):
    y_pred = np.array([np.inf, 0, 0])
    y_true = np.array([1, 0, 0])
    for objective in OPTIONS.values():
        out, err = capsys.readouterr()
        objective.score(y_true, y_pred)
        assert out == "WARNING: y_predicted contains NaN or infinity"
        assert err == ""
    y_true = np.array([np.inf, 0, 0])
    y_pred = np.array([1, 0, 0])
    for objective in OPTIONS.values():
        out, err = capsys.readouterr()
        objective.score(y_true, y_pred)
        assert out == "WARNING: y_true contains infinity values"
        assert err == ""


def test_different_input_lengths():
    y_pred = np.array([0, 0])
    y_true = np.array([1])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
            objective.score(y_true, y_pred)

    y_true = np.array([0, 0])
    y_pred = np.array([1, 2, 0])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
            objective.score(y_true, y_pred)


def test_zero_input_lengths():
    y_pred = np.array([])
    y_true = np.array([])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="Length of inputs is 0"):
            objective.score(y_true, y_pred)


def test_probabilities_not_in_0_1_range():
    y_pred = np.array([0.3, 1.001, 0.3])
    y_true = np.array([1, 0, 1])
    for objective in OPTIONS.values():
        if objective.score_needs_proba:
            with pytest.raises(ValueError, match="y_predicted contains probability estimates"):
                objective.score(y_true, y_pred)

    y_pred = np.array([0.3, -0.001, 0.3])
    y_true = np.array([1, 0, 1])
    for objective in OPTIONS.values():
        if objective.score_needs_proba:
            with pytest.raises(ValueError, match="y_predicted contains probability estimates"):
                objective.score(y_true, y_pred)


def test_binary_more_than_two_unique_values():
    y_pred = np.array([0, 1, 2])
    y_true = np.array([1, 0, 1])
    for objective in OPTIONS.values():
        if isinstance(objective, BinaryClassificationObjective) and not objective.score_needs_proba:
            with pytest.raises(ValueError, match="y_predicted contains more than two unique values"):
                objective.score(y_true, y_pred)

    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 0, 1])
    for objective in OPTIONS.values():
        if isinstance(objective, BinaryClassificationObjective) and not objective.score_needs_proba:
            with pytest.raises(ValueError, match="y_true contains more than two unique values"):
                objective.score(y_true, y_pred)


def test_accuracy_binary():
    obj = AccuracyBinary()
    assert obj.score(np.array([0, 0, 1, 1]),
                     np.array([1, 1, 0, 0])) == pytest.approx(0.0, EPS)
    assert obj.score(np.array([0, 0, 1, 1]),
                     np.array([0, 1, 0, 1])) == pytest.approx(0.5, EPS)
    assert obj.score(np.array([0, 0, 1, 1]),
                     np.array([0, 0, 1, 1])) == pytest.approx(1.0, EPS)


def test_accuracy_multi():
    obj = AccuracyMulticlass()
    assert obj.score(np.array([0, 0, 1, 1]),
                     np.array([1, 1, 0, 0])) == pytest.approx(0.0, EPS)
    assert obj.score(np.array([0, 0, 1, 1]),
                     np.array([0, 1, 0, 1])) == pytest.approx(0.5, EPS)
    assert obj.score(np.array([0, 0, 1, 1]),
                     np.array([0, 0, 1, 1])) == pytest.approx(1.0, EPS)
    assert obj.score(np.array([0, 0, 1, 1, 2, 2]),
                     np.array([0, 0, 0, 0, 0, 0])) == pytest.approx(1 / 3.0, EPS)
    assert obj.score(np.array([0, 0, 0, 0, 0, 0]),
                     np.array([0, 0, 1, 1, 2, 2])) == pytest.approx(1 / 3.0, EPS)


def test_balanced_accuracy_binary():
    obj = BalancedAccuracyBinary()
    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.625, EPS)

    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 1, 0])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([1, 0, 1, 1, 0, 1])) == pytest.approx(0.0, EPS)


def test_balanced_accuracy_multi():
    obj = BalancedAccuracyMulticlass()
    assert obj.score(np.array([0, 1, 2, 0, 1, 2, 3]),
                     np.array([0, 0, 2, 0, 0, 2, 3])) == pytest.approx(0.75, EPS)

    assert obj.score(np.array([0, 1, 2, 0, 1, 2, 3]),
                     np.array([0, 1, 2, 0, 1, 2, 3])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 1, 2, 0, 1, 2, 3]),
                     np.array([1, 0, 3, 1, 2, 1, 0])) == pytest.approx(0.0, EPS)


def test_f1_binary():
    obj = F1()
    assert obj.score(np.array([0, 1, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.5, EPS)

    assert obj.score(np.array([0, 1, 0, 0, 1, 1]),
                     np.array([0, 1, 0, 0, 1, 1])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 0, 1, 0]),
                     np.array([0, 1, 0, 0, 0, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([0, 0])) == pytest.approx(0.0, EPS)


def test_f1_micro_multi():
    obj = F1Micro()
    assert obj.score(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([2, 2, 2, 0, 0, 0, 1, 1, 1]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([1, 2]),
                     np.array([0, 0])) == pytest.approx(0.0, EPS)


def test_f1_macro_multi():
    obj = F1Macro()
    assert obj.score(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) \
        == pytest.approx(2 * (1 / 3.0) * (1 / 9.0) / (1 / 3.0 + 1 / 9.0), EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([2, 2, 2, 0, 0, 0, 1, 1, 1]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([1, 2]),
                     np.array([0, 0])) == pytest.approx(0.0, EPS)


def test_f1_weighted_multi():
    obj = F1Weighted()
    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) \
        == pytest.approx(2 * (1 / 3.0) * (1 / 9.0) / (1 / 3.0 + 1 / 9.0), EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 2])) == pytest.approx(0.0, EPS)


def test_precision_binary():
    obj = Precision()
    assert obj.score(np.array([1, 1, 1, 1, 1, 1]),
                     np.array([0, 0, 0, 1, 1, 1])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1]),
                     np.array([1, 1, 1, 1, 1, 1])) == pytest.approx(0.5, EPS)

    assert obj.score(np.array([0, 0, 0, 0, 0, 0]),
                     np.array([1, 1, 1, 1, 1, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0, 0, 0, 0, 0]),
                     np.array([0, 0, 0, 0, 0, 0])) == pytest.approx(0.0, EPS)


def test_precision_micro_multi():
    obj = PrecisionMicro()
    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 2])) == pytest.approx(0.0, EPS)


def test_precision_macro_multi():
    obj = PrecisionMacro()
    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) == pytest.approx(1 / 9.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 2])) == pytest.approx(0.0, EPS)


def test_precision_weighted_multi():
    obj = PrecisionWeighted()
    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) == pytest.approx(1 / 9.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 2])) == pytest.approx(0.0, EPS)


def test_recall_binary():
    obj = Recall()
    assert obj.score(np.array([0, 0, 0, 1, 1, 1]),
                     np.array([1, 1, 1, 1, 1, 1])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1]),
                     np.array([0, 0, 0, 0, 0, 0])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([1, 1, 1, 1, 1, 1]),
                     np.array([0, 0, 0, 1, 1, 1])) == pytest.approx(0.5, EPS)


def test_recall_micro_multi():
    obj = RecallMicro()
    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 2])) == pytest.approx(0.0, EPS)


def test_recall_macro_multi():
    obj = RecallMacro()
    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 2])) == pytest.approx(0.0, EPS)


def test_recall_weighted_multi():
    obj = RecallWeighted()
    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])) == pytest.approx(1 / 3.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])) == pytest.approx(1.0, EPS)

    assert obj.score(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                     np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])) == pytest.approx(0.0, EPS)

    assert obj.score(np.array([0, 0]),
                     np.array([1, 2])) == pytest.approx(0.0, EPS)
