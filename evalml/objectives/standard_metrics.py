"""Standard machine learning objective functions."""
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from evalml.objectives.binary_classification_objective import (
    BinaryClassificationObjective,
)
from evalml.objectives.multiclass_classification_objective import (
    MulticlassClassificationObjective,
)
from evalml.objectives.regression_objective import RegressionObjective
from evalml.objectives.time_series_regression_objective import (
    TimeSeriesRegressionObjective,
)
from evalml.utils import classproperty


class AccuracyBinary(BinaryClassificationObjective):
    """Accuracy score for binary classification.

    Example:
        >>> y_true = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(AccuracyBinary().objective_function(y_true, y_pred), 0.6363636)
    """

    name = "Accuracy Binary"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for accuracy score for binary classification."""
        return metrics.accuracy_score(y_true, y_predicted, sample_weight=sample_weight)


class AccuracyMulticlass(MulticlassClassificationObjective):
    """Accuracy score for multiclass classification.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(AccuracyMulticlass().objective_function(y_true, y_pred), 0.5454545)
    """

    name = "Accuracy Multiclass"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for accuracy score for multiclass classification."""
        return metrics.accuracy_score(y_true, y_predicted, sample_weight=sample_weight)


class BalancedAccuracyBinary(BinaryClassificationObjective):
    """Balanced accuracy score for binary classification.

    Example:
        >>> y_true = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(BalancedAccuracyBinary().objective_function(y_true, y_pred), 0.60)
    """

    name = "Balanced Accuracy Binary"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for accuracy score for balanced accuracy for binary classification."""
        return metrics.balanced_accuracy_score(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )


class BalancedAccuracyMulticlass(MulticlassClassificationObjective):
    """Balanced accuracy score for multiclass classification.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(BalancedAccuracyMulticlass().objective_function(y_true, y_pred), 0.5555555)
    """

    name = "Balanced Accuracy Multiclass"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for accuracy score for balanced accuracy for multiclass classification."""
        return metrics.balanced_accuracy_score(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )


class F1(BinaryClassificationObjective):
    """F1 score for binary classification.

    Example:
        >>> y_true = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(F1().objective_function(y_true, y_pred), 0.25)
    """

    name = "F1"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for F1 score for binary classification."""
        return metrics.f1_score(
            y_true,
            y_predicted,
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class F1Micro(MulticlassClassificationObjective):
    """F1 score for multiclass classification using micro averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(F1Micro().objective_function(y_true, y_pred), 0.5454545)
    """

    name = "F1 Micro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for F1 score for multiclass classification."""
        return metrics.f1_score(
            y_true,
            y_predicted,
            average="micro",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class F1Macro(MulticlassClassificationObjective):
    """F1 score for multiclass classification using macro averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(F1Macro().objective_function(y_true, y_pred), 0.5476190)
    """

    name = "F1 Macro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for F1 score for multiclass classification using macro averaging."""
        return metrics.f1_score(
            y_true,
            y_predicted,
            average="macro",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class F1Weighted(MulticlassClassificationObjective):
    """F1 score for multiclass classification using weighted averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(F1Weighted().objective_function(y_true, y_pred), 0.5454545)
    """

    name = "F1 Weighted"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for F1 score for multiclass classification using weighted averaging."""
        return metrics.f1_score(
            y_true,
            y_predicted,
            average="weighted",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class Precision(BinaryClassificationObjective):
    """Precision score for binary classification.

    Example:
        >>> y_true = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(Precision().objective_function(y_true, y_pred), 1.0)
    """

    name = "Precision"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for precision score for binary classification."""
        return metrics.precision_score(
            y_true,
            y_predicted,
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class PrecisionMicro(MulticlassClassificationObjective):
    """Precision score for multiclass classification using micro averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(PrecisionMicro().objective_function(y_true, y_pred), 0.5454545)
    """

    name = "Precision Micro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for precision score for binary classification using micro-averaging."""
        return metrics.precision_score(
            y_true,
            y_predicted,
            average="micro",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class PrecisionMacro(MulticlassClassificationObjective):
    """Precision score for multiclass classification using macro-averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(PrecisionMacro().objective_function(y_true, y_pred), 0.5555555)
    """

    name = "Precision Macro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for precision score for multiclass classification using macro-averaging."""
        return metrics.precision_score(
            y_true,
            y_predicted,
            average="macro",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class PrecisionWeighted(MulticlassClassificationObjective):
    """Precision score for multiclass classification using weighted averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(PrecisionWeighted().objective_function(y_true, y_pred), 0.5606060)
    """

    name = "Precision Weighted"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for precision score for multiclass classification using weighted averaging."""
        return metrics.precision_score(
            y_true,
            y_predicted,
            average="weighted",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class Recall(BinaryClassificationObjective):
    """Recall score for binary classification.

    Example:
        >>> y_true = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(Recall().objective_function(y_true, y_pred), 0.1428571)
    """

    name = "Recall"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for recall score for binary classification."""
        return metrics.recall_score(
            y_true,
            y_predicted,
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class RecallMicro(MulticlassClassificationObjective):
    """Recall score for multiclass classification using micro averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(RecallMicro().objective_function(y_true, y_pred), 0.5454545)
    """

    name = "Recall Micro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for recall score for multiclass classification using micro-averaging."""
        return metrics.recall_score(
            y_true,
            y_predicted,
            average="micro",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class RecallMacro(MulticlassClassificationObjective):
    """Recall score for multiclass classification using macro averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(RecallMacro().objective_function(y_true, y_pred), 0.5555555)
    """

    name = "Recall Macro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for recall score for multiclass classification using macro-averaging."""
        return metrics.recall_score(
            y_true,
            y_predicted,
            average="macro",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class RecallWeighted(MulticlassClassificationObjective):
    """Recall score for multiclass classification using weighted averaging.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(RecallWeighted().objective_function(y_true, y_pred), 0.5454545)
    """

    name = "Recall Weighted"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for recall score for multiclass classification using weighted averaging."""
        return metrics.recall_score(
            y_true,
            y_predicted,
            average="weighted",
            zero_division=0.0,
            sample_weight=sample_weight,
        )


class AUC(BinaryClassificationObjective):
    """AUC score for binary classification.

    Example:
        >>> y_true = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(AUC().objective_function(y_true, y_pred), 0.5714285)
    """

    name = "AUC"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for AUC score for binary classification."""
        return metrics.roc_auc_score(y_true, y_predicted, sample_weight=sample_weight)


class AUCMicro(MulticlassClassificationObjective):
    """AUC score for multiclass classification using micro averaging.

    Example:
        >>> y_true = [0, 1, 2, 0, 2, 1]
        >>> y_pred = [[0.7, 0.2, 0.1],
        ...           [0.3, 0.5, 0.2],
        ...           [0.1, 0.3, 0.6],
        ...           [0.9, 0.1, 0.0],
        ...           [0.3, 0.1, 0.6],
        ...           [0.5, 0.5, 0.0]]
        >>> np.testing.assert_almost_equal(AUCMicro().objective_function(y_true, y_pred), 0.9861111)
    """

    name = "AUC Micro"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for AUC score for multiclass classification using micro-averaging."""
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(
            y_true,
            y_predicted,
            average="micro",
            sample_weight=sample_weight,
        )


class AUCMacro(MulticlassClassificationObjective):
    """AUC score for multiclass classification using macro averaging.

    Example:
        >>> y_true = [0, 1, 2, 0, 2, 1]
        >>> y_pred = [[0.7, 0.2, 0.1],
        ...           [0.1, 0.0, 0.9],
        ...           [0.1, 0.3, 0.6],
        ...           [0.9, 0.1, 0.0],
        ...           [0.6, 0.1, 0.3],
        ...           [0.5, 0.5, 0.0]]
        >>> np.testing.assert_almost_equal(AUCMacro().objective_function(y_true, y_pred), 0.75)
    """

    name = "AUC Macro"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for AUC score for multiclass classification using macro-averaging."""
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(
            y_true,
            y_predicted,
            average="macro",
            sample_weight=sample_weight,
        )


class AUCWeighted(MulticlassClassificationObjective):
    """AUC Score for multiclass classification using weighted averaging.

    Example:
        >>> y_true = [0, 1, 2, 0, 2, 1]
        >>> y_pred = [[0.7, 0.2, 0.1],
        ...           [0.1, 0.0, 0.9],
        ...           [0.1, 0.3, 0.6],
        ...           [0.1, 0.2, 0.7],
        ...           [0.6, 0.1, 0.3],
        ...           [0.5, 0.2, 0.3]]
        >>> np.testing.assert_almost_equal(AUCWeighted().objective_function(y_true, y_pred), 0.4375)
    """

    name = "AUC Weighted"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for AUC Score for multiclass classification using weighted averaging."""
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(
            y_true,
            y_predicted,
            average="weighted",
            sample_weight=sample_weight,
        )


class Gini(BinaryClassificationObjective):
    """Gini coefficient for binary classification.

    Example:
        >>> y_true = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(Gini().objective_function(y_true, y_pred), 0.1428571)
    """

    name = "Gini"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = False
    expected_range = [-1, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for Gini coefficient for binary classification."""
        auc = metrics.roc_auc_score(y_true, y_predicted, sample_weight=sample_weight)
        return 2 * auc - 1


class LogLossBinary(BinaryClassificationObjective):
    """Log Loss for binary classification.

    Example:
        >>> y_true = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(LogLossBinary().objective_function(y_true, y_pred), 18.8393325)
    """

    name = "Log Loss Binary"
    greater_is_better = False
    score_needs_proba = True
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for log loss for binary classification."""
        return metrics.log_loss(y_true, y_predicted, sample_weight=sample_weight)


class LogLossMulticlass(MulticlassClassificationObjective):
    """Log Loss for multiclass classification.

    Example:
        >>> y_true = [0, 1, 2, 0, 2, 1]
        >>> y_pred = [[0.7, 0.2, 0.1],
        ...           [0.3, 0.5, 0.2],
        ...           [0.1, 0.3, 0.6],
        ...           [0.9, 0.1, 0.0],
        ...           [0.3, 0.1, 0.6],
        ...           [0.5, 0.5, 0.0]]
        >>> np.testing.assert_almost_equal(LogLossMulticlass().objective_function(y_true, y_pred), 0.4783301)
    """

    name = "Log Loss Multiclass"
    greater_is_better = False
    score_needs_proba = True
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for log loss for multiclass classification."""
        return metrics.log_loss(y_true, y_predicted, sample_weight=sample_weight)


class MCCBinary(BinaryClassificationObjective):
    """Matthews correlation coefficient for binary classification.

    Example:
        >>> y_true = pd.Series([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        >>> y_pred = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        >>> np.testing.assert_almost_equal(MCCBinary().objective_function(y_true, y_pred), 0.2390457)
    """

    name = "MCC Binary"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False  # Range [-1, 1]
    expected_range = [-1, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for Matthews correlation coefficient for binary classification."""
        with warnings.catch_warnings():
            # catches runtime warning when dividing by 0.0
            warnings.simplefilter("ignore", RuntimeWarning)
            return metrics.matthews_corrcoef(
                y_true,
                y_predicted,
                sample_weight=sample_weight,
            )


class MCCMulticlass(MulticlassClassificationObjective):
    """Matthews correlation coefficient for multiclass classification.

    Example:
        >>> y_true = pd.Series([0, 1, 0, 2, 0, 1, 2, 1, 2, 0, 2])
        >>> y_pred = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        >>> np.testing.assert_almost_equal(MCCMulticlass().objective_function(y_true, y_pred), 0.325)
    """

    name = "MCC Multiclass"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False  # Range [-1, 1]
    expected_range = [-1, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for Matthews correlation coefficient for multiclass classification."""
        with warnings.catch_warnings():
            # catches runtime warning when dividing by 0.0
            warnings.simplefilter("ignore", RuntimeWarning)
            return metrics.matthews_corrcoef(
                y_true,
                y_predicted,
                sample_weight=sample_weight,
            )


class RootMeanSquaredError(RegressionObjective):
    """Root mean squared error for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(RootMeanSquaredError().objective_function(y_true, y_pred), 0.3988620)
    """

    name = "Root Mean Squared Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for root mean squared error for regression."""
        return metrics.mean_squared_error(
            y_true,
            y_predicted,
            squared=False,
            sample_weight=sample_weight,
        )


class RootMeanSquaredLogError(RegressionObjective):
    """Root mean squared log error for regression.

    Only valid for nonnegative inputs. Otherwise, will throw a ValueError.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(RootMeanSquaredLogError().objective_function(y_true, y_pred), 0.13090204)
    """

    name = "Root Mean Squared Log Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for root mean squared log error for regression."""
        return np.sqrt(
            metrics.mean_squared_log_error(
                y_true,
                y_predicted,
                sample_weight=sample_weight,
            ),
        )

    @classproperty
    def positive_only(self):
        """If True, this objective is only valid for positive data."""
        return True


class MeanSquaredLogError(RegressionObjective):
    """Mean squared log error for regression.

    Only valid for nonnegative inputs. Otherwise, will throw a ValueError.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(MeanSquaredLogError().objective_function(y_true, y_pred), 0.0171353)
    """

    name = "Mean Squared Log Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for mean squared log error for regression."""
        return metrics.mean_squared_log_error(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )

    @classproperty
    def positive_only(self):
        """If True, this objective is only valid for positive data."""
        return True


class R2(RegressionObjective):
    """Coefficient of determination for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(R2().objective_function(y_true, y_pred), 0.7638036)
    """

    name = "R2"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1
    is_bounded_like_percentage = False  # Range (-Inf, 1]
    expected_range = [-1, 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for coefficient of determination for regression."""
        return metrics.r2_score(y_true, y_predicted, sample_weight=sample_weight)


class MAE(RegressionObjective):
    """Mean absolute error for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(MAE().objective_function(y_true, y_pred), 0.2727272)
    """

    name = "MAE"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = True  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for mean absolute error for regression."""
        return metrics.mean_absolute_error(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )


class MAPE(TimeSeriesRegressionObjective):
    """Mean absolute percentage error for time series regression. Scaled by 100 to return a percentage.

    Only valid for nonzero inputs. Otherwise, will throw a ValueError.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(MAPE().objective_function(y_true, y_pred), 15.9848484)
    """

    name = "Mean Absolute Percentage Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for mean absolute percentage error for time series regression."""
        if (y_true == 0).any():
            raise ValueError(
                "Mean Absolute Percentage Error cannot be used when "
                "targets contain the value 0.",
            )
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_predicted, pd.Series):
            y_predicted = y_predicted.values
        scaled_difference = (y_true - y_predicted) / y_true
        return np.abs(scaled_difference).mean() * 100

    @classproperty
    def positive_only(self):
        """If True, this objective is only valid for positive data."""
        return True


class MSE(RegressionObjective):
    """Mean squared error for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(MSE().objective_function(y_true, y_pred), 0.1590909)
    """

    name = "MSE"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for mean squared error for regression."""
        return metrics.mean_squared_error(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )


class MedianAE(RegressionObjective):
    """Median absolute error for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(MedianAE().objective_function(y_true, y_pred), 0.25)
    """

    name = "MedianAE"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for median absolute error for regression."""
        return metrics.median_absolute_error(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )


class MaxError(RegressionObjective):
    """Maximum residual error for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(MaxError().objective_function(y_true, y_pred), 1.0)
    """

    name = "MaxError"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)
    expected_range = [0, float("inf")]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for maximum residual error for regression."""
        return metrics.max_error(y_true, y_predicted)


class ExpVariance(RegressionObjective):
    """Explained variance score for regression.

    Example:
        >>> y_true = pd.Series([1.5, 2, 3, 1, 0.5, 1, 2.5, 2.5, 1, 0.5, 2])
        >>> y_pred = pd.Series([1.5, 2.5, 2, 1, 0.5, 1, 3, 2.25, 0.75, 0.25, 1.75])
        >>> np.testing.assert_almost_equal(ExpVariance().objective_function(y_true, y_pred), 0.7760736)
    """

    name = "ExpVariance"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False  # Range (-Inf, 1]
    expected_range = [float("-inf"), 1]

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Objective function for explained variance score for regression."""
        return metrics.explained_variance_score(
            y_true,
            y_predicted,
            sample_weight=sample_weight,
        )


def _handle_predictions(y_true, y_pred):
    if len(np.unique(y_true)) > 2:
        classes = np.unique(y_true)
        y_true = label_binarize(y_true, classes=classes)

    return y_true, y_pred
