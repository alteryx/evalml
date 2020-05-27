import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef as sk_matthews_corrcoef

from evalml.objectives import (
    F1,
    MSE,
    AccuracyBinary,
    AccuracyMulticlass,
    BalancedAccuracyBinary,
    BalancedAccuracyMulticlass,
    BinaryClassificationObjective,
    F1Macro,
    F1Micro,
    F1Weighted,
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
    RootMeanSquaredLogError
)
from evalml.objectives.utils import OPTIONS

EPS = 1e-5


def test_input_contains_nan():
    y_predicted = np.array([np.nan, 0, 0])
    y_true = np.array([1, 2, 1])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
            objective.score(y_true, y_predicted)

    y_true = np.array([np.nan, 0, 0])
    y_predicted = np.array([1, 2, 0])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
            objective.score(y_true, y_predicted)


def test_input_contains_inf():
    y_predicted = np.array([np.inf, 0, 0])
    y_true = np.array([1, 0, 0])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
            objective.score(y_true, y_predicted)

    y_true = np.array([np.inf, 0, 0])
    y_predicted = np.array([1, 0, 0])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="y_true contains NaN or infinity"):
            objective.score(y_true, y_predicted)


def test_different_input_lengths():
    y_predicted = np.array([0, 0])
    y_true = np.array([1])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
            objective.score(y_true, y_predicted)

    y_true = np.array([0, 0])
    y_predicted = np.array([1, 2, 0])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
            objective.score(y_true, y_predicted)


def test_zero_input_lengths():
    y_predicted = np.array([])
    y_true = np.array([])
    for objective in OPTIONS.values():
        with pytest.raises(ValueError, match="Length of inputs is 0"):
            objective.score(y_true, y_predicted)


def test_probabilities_not_in_0_1_range():
    y_predicted = np.array([0.3, 1.001, 0.3])
    y_true = np.array([1, 0, 1])
    for objective in OPTIONS.values():
        if objective.score_needs_proba:
            with pytest.raises(ValueError, match="y_predicted contains probability estimates"):
                objective.score(y_true, y_predicted)

    y_predicted = np.array([0.3, -0.001, 0.3])
    y_true = np.array([1, 0, 1])
    for objective in OPTIONS.values():
        if objective.score_needs_proba:
            with pytest.raises(ValueError, match="y_predicted contains probability estimates"):
                objective.score(y_true, y_predicted)


def test_negative_with_log():
    y_predicted = np.array([-1, 10, 30])
    y_true = np.array([-1, 0, 1])
    for objective in [MeanSquaredLogError(), RootMeanSquaredLogError()]:
        with pytest.raises(ValueError, match="Mean Squared Logarithmic Error cannot be used when targets contain negative values."):
            objective.score(y_true, y_predicted)


def test_binary_more_than_two_unique_values():
    y_predicted = np.array([0, 1, 2])
    y_true = np.array([1, 0, 1])
    for objective in OPTIONS.values():
        if isinstance(objective, BinaryClassificationObjective) and not objective.score_needs_proba:
            with pytest.raises(ValueError, match="y_predicted contains more than two unique values"):
                objective.score(y_true, y_predicted)

    y_true = np.array([0, 1, 2])
    y_predicted = np.array([1, 0, 1])
    for objective in OPTIONS.values():
        if isinstance(objective, BinaryClassificationObjective) and not objective.score_needs_proba:
            with pytest.raises(ValueError, match="y_true contains more than two unique values"):
                objective.score(y_true, y_predicted)


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


def test_log_linear_model():
    obj = MeanSquaredLogError()
    root_obj = RootMeanSquaredLogError()

    s1_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s1_actual = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    s2_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s2_actual = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    s3_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s3_actual = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])

    assert obj.score(s1_predicted, s1_actual) == pytest.approx(0.562467324910)
    assert obj.score(s2_predicted, s2_actual) == pytest.approx(0)
    assert obj.score(s3_predicted, s3_actual) == pytest.approx(0.617267976207983)

    assert root_obj.score(s1_predicted, s1_actual) == pytest.approx(np.sqrt(0.562467324910))
    assert root_obj.score(s2_predicted, s2_actual) == pytest.approx(0)
    assert root_obj.score(s3_predicted, s3_actual) == pytest.approx(np.sqrt(0.617267976207983))


def test_mse_linear_model():
    obj = MSE()
    root_obj = RootMeanSquaredError()

    s1_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s1_actual = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    s2_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s2_actual = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    s3_predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    s3_actual = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])

    assert obj.score(s1_predicted, s1_actual) == pytest.approx(5. / 3.)
    assert obj.score(s2_predicted, s2_actual) == pytest.approx(0)
    assert obj.score(s3_predicted, s3_actual) == pytest.approx(2.)

    assert root_obj.score(s1_predicted, s1_actual) == pytest.approx(np.sqrt(5. / 3.))
    assert root_obj.score(s2_predicted, s2_actual) == pytest.approx(0)
    assert root_obj.score(s3_predicted, s3_actual) == pytest.approx(np.sqrt(2.))


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
