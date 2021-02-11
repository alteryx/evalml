import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from ..utils import classproperty
from .binary_classification_objective import BinaryClassificationObjective
from .multiclass_classification_objective import (
    MulticlassClassificationObjective
)
from .regression_objective import RegressionObjective
from .time_series_regression_objective import TimeSeriesRegressionObjective


class AccuracyBinary(BinaryClassificationObjective):
    """Accuracy score for binary classification."""
    name = "Accuracy Binary"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.accuracy_score(y_true, y_predicted)


class AccuracyMulticlass(MulticlassClassificationObjective):
    """Accuracy score for multiclass classification."""
    name = "Accuracy Multiclass"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.accuracy_score(y_true, y_predicted)


class BalancedAccuracyBinary(BinaryClassificationObjective):
    """Balanced accuracy score for binary classification."""
    name = "Balanced Accuracy Binary"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.balanced_accuracy_score(y_true, y_predicted)


class BalancedAccuracyMulticlass(MulticlassClassificationObjective):
    """Balanced accuracy score for multiclass classification."""
    name = "Balanced Accuracy Multiclass"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.balanced_accuracy_score(y_true, y_predicted)


class F1(BinaryClassificationObjective):
    """F1 score for binary classification."""
    name = "F1"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.f1_score(y_true, y_predicted, zero_division=0.0)


class F1Micro(MulticlassClassificationObjective):
    """F1 score for multiclass classification using micro averaging."""
    name = "F1 Micro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.f1_score(y_true, y_predicted, average='micro', zero_division=0.0)


class F1Macro(MulticlassClassificationObjective):
    """F1 score for multiclass classification using macro averaging."""
    name = "F1 Macro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.f1_score(y_true, y_predicted, average='macro', zero_division=0.0)


class F1Weighted(MulticlassClassificationObjective):
    """F1 score for multiclass classification using weighted averaging."""
    name = "F1 Weighted"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.f1_score(y_true, y_predicted, average='weighted', zero_division=0.0)


class Precision(BinaryClassificationObjective):
    """Precision score for binary classification."""
    name = "Precision"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.precision_score(y_true, y_predicted, zero_division=0.0)


class PrecisionMicro(MulticlassClassificationObjective):
    """Precision score for multiclass classification using micro averaging."""
    name = "Precision Micro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.precision_score(y_true, y_predicted, average='micro', zero_division=0.0)


class PrecisionMacro(MulticlassClassificationObjective):
    """Precision score for multiclass classification using macro averaging."""
    name = "Precision Macro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.precision_score(y_true, y_predicted, average='macro', zero_division=0.0)


class PrecisionWeighted(MulticlassClassificationObjective):
    """Precision score for multiclass classification using weighted averaging."""
    name = "Precision Weighted"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.precision_score(y_true, y_predicted, average='weighted', zero_division=0.0)


class Recall(BinaryClassificationObjective):
    """Recall score for binary classification."""
    name = "Recall"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.recall_score(y_true, y_predicted, zero_division=0.0)


class RecallMicro(MulticlassClassificationObjective):
    """Recall score for multiclass classification using micro averaging."""
    name = "Recall Micro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.recall_score(y_true, y_predicted, average='micro', zero_division=0.0)


class RecallMacro(MulticlassClassificationObjective):
    """Recall score for multiclass classification using macro averaging."""
    name = "Recall Macro"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.recall_score(y_true, y_predicted, average='macro', zero_division=0.0)


class RecallWeighted(MulticlassClassificationObjective):
    """Recall score for multiclass classification using weighted averaging."""
    name = "Recall Weighted"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.recall_score(y_true, y_predicted, average='weighted', zero_division=0.0)


class AUC(BinaryClassificationObjective):
    """AUC score for binary classification."""
    name = "AUC"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.roc_auc_score(y_true, y_predicted)


class AUCMicro(MulticlassClassificationObjective):
    """AUC score for multiclass classification using micro averaging."""
    name = "AUC Micro"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='micro')


class AUCMacro(MulticlassClassificationObjective):
    """AUC score for multiclass classification using macro averaging."""
    name = "AUC Macro"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='macro')


class AUCWeighted(MulticlassClassificationObjective):
    """AUC Score for multiclass classification using weighted averaging."""
    name = "AUC Weighted"
    greater_is_better = True
    score_needs_proba = True
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def objective_function(self, y_true, y_predicted, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='weighted')


class LogLossBinary(BinaryClassificationObjective):
    """Log Loss for binary classification."""
    name = "Log Loss Binary"
    greater_is_better = False
    score_needs_proba = True
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.log_loss(y_true, y_predicted)


class LogLossMulticlass(MulticlassClassificationObjective):
    """Log Loss for multiclass classification."""
    name = "Log Loss Multiclass"
    greater_is_better = False
    score_needs_proba = True
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.log_loss(y_true, y_predicted)


class MCCBinary(BinaryClassificationObjective):
    """Matthews correlation coefficient for binary classification."""
    name = "MCC Binary"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False  # Range [-1, 1]1

    def objective_function(self, y_true, y_predicted, X=None):
        with warnings.catch_warnings():
            # catches runtime warning when dividing by 0.0
            warnings.simplefilter('ignore', RuntimeWarning)
            return metrics.matthews_corrcoef(y_true, y_predicted)


class MCCMulticlass(MulticlassClassificationObjective):
    """Matthews correlation coefficient for multiclass classification."""
    name = "MCC Multiclass"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False  # Range [-1, 1]

    def objective_function(self, y_true, y_predicted, X=None):
        with warnings.catch_warnings():
            # catches runtime warning when dividing by 0.0
            warnings.simplefilter('ignore', RuntimeWarning)
            return metrics.matthews_corrcoef(y_true, y_predicted)


class RootMeanSquaredError(RegressionObjective):
    """Root mean squared error for regression."""
    name = "Root Mean Squared Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.mean_squared_error(y_true, y_predicted, squared=False)


class RootMeanSquaredLogError(RegressionObjective):
    """Root mean squared log error for regression.

    Only valid for nonnegative inputs.Otherwise, will throw a ValueError.
    """
    name = "Root Mean Squared Log Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return np.sqrt(metrics.mean_squared_log_error(y_true, y_predicted))

    @classproperty
    def positive_only(self):
        """If True, this objective is only valid for positive data. Default False."""
        return True


class MeanSquaredLogError(RegressionObjective):
    """Mean squared log error for regression.

    Only valid for nonnegative inputs. Otherwise, will throw a ValueError
    """
    name = "Mean Squared Log Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.mean_squared_log_error(y_true, y_predicted)

    @classproperty
    def positive_only(self):
        """If True, this objective is only valid for positive data. Default False."""
        return True


class R2(RegressionObjective):
    """Coefficient of determination for regression."""
    name = "R2"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1
    is_bounded_like_percentage = False  # Range (-Inf, 1]

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.r2_score(y_true, y_predicted)


class MAE(RegressionObjective):
    """Mean absolute error for regression."""
    name = "MAE"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = True  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.mean_absolute_error(y_true, y_predicted)


class MAPE(TimeSeriesRegressionObjective):
    """Mean absolute percentage error for time series regression. Scaled by 100 to return a percentage.

    Only valid for nonzero inputs. Otherwise, will throw a ValueError
    """
    name = "Mean Absolute Percentage Error"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        if (y_true == 0).any():
            raise ValueError("Mean Absolute Percentage Error cannot be used when "
                             "targets contain the value 0.")
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_predicted, pd.Series):
            y_predicted = y_predicted.values
        scaled_difference = (y_true - y_predicted) / y_true
        return np.abs(scaled_difference).mean() * 100

    @classproperty
    def positive_only(self):
        """If True, this objective is only valid for positive data. Default False."""
        return True


class MSE(RegressionObjective):
    """Mean squared error for regression."""
    name = "MSE"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.mean_squared_error(y_true, y_predicted)


class MedianAE(RegressionObjective):
    """Median absolute error for regression."""
    name = "MedianAE"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.median_absolute_error(y_true, y_predicted)


class MaxError(RegressionObjective):
    """Maximum residual error for regression."""
    name = "MaxError"
    greater_is_better = False
    score_needs_proba = False
    perfect_score = 0.0
    is_bounded_like_percentage = False  # Range [0, Inf)

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.max_error(y_true, y_predicted)


class ExpVariance(RegressionObjective):
    """Explained variance score for regression."""
    name = "ExpVariance"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = False  # Range (-Inf, 1]

    def objective_function(self, y_true, y_predicted, X=None):
        return metrics.explained_variance_score(y_true, y_predicted)


def _handle_predictions(y_true, y_pred):
    if len(np.unique(y_true)) > 2:
        classes = np.unique(y_true)
        y_true = label_binarize(y_true, classes=classes)

    return y_true, y_pred
