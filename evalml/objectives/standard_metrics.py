from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

from .binary_classification_objective import BinaryClassificationObjective
from .multiclass_classification_objective import MultiClassificationObjective
from .regression_objective import RegressionObjective


# todo does this need tuning?
class F1(BinaryClassificationObjective):
    """F1 score for binary classification"""
    greater_is_better = True
    name = "F1"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.f1_score(y_true, y_predicted)


class F1Micro(MultiClassificationObjective):
    """F1 score for multiclass classification using micro averaging"""
    greater_is_better = True
    name = "F1 Micro"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.f1_score(y_true, y_predicted, average='micro')


class F1Macro(MultiClassificationObjective):
    """F1 score for multiclass classification using macro averaging"""
    greater_is_better = True
    name = "F1 Macro"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.f1_score(y_true, y_predicted, average='macro')


class F1Weighted(MultiClassificationObjective):
    """F1 score for multiclass classification using weighted averaging"""
    greater_is_better = True
    name = "F1 Weighted"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.f1_score(y_true, y_predicted, average='weighted')


class Precision(BinaryClassificationObjective):
    """Precision score for binary classification"""
    greater_is_better = True
    name = "Precision"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.precision_score(y_true, y_predicted)


class PrecisionMicro(BinaryClassificationObjective):
    """Precision score for multiclass classification using micro averaging"""
    greater_is_better = True
    name = "Precision Micro"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.precision_score(y_true, y_predicted, average='micro')


class PrecisionMacro(BinaryClassificationObjective):
    """Precision score for multiclass classification using macro averaging"""
    greater_is_better = True
    name = "Precision Macro"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.precision_score(y_true, y_predicted, average='macro')


class PrecisionWeighted(MultiClassificationObjective):
    """Precision score for multiclass classification using weighted averaging"""
    greater_is_better = True
    name = "Precision Weighted"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.precision_score(y_true, y_predicted, average='weighted')


class Recall(BinaryClassificationObjective):
    """Recall score for binary classification"""
    greater_is_better = True
    name = "Recall"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.recall_score(y_true, y_predicted)


class RecallMicro(MultiClassificationObjective):
    """Recall score for multiclass classification using micro averaging"""
    greater_is_better = True
    name = "Recall Micro"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.recall_score(y_true, y_predicted, average='micro')


class RecallMacro(MultiClassificationObjective):
    """Recall score for multiclass classification using macro averaging"""
    greater_is_better = True
    name = "Recall Macro"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.recall_score(y_true, y_predicted, average='macro')


class RecallWeighted(MultiClassificationObjective):
    """Recall score for multiclass classification using weighted averaging"""
    greater_is_better = True
    name = "Recall Weighted"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.recall_score(y_true, y_predicted, average='weighted')


class AUC(BinaryClassificationObjective):
    """AUC score for binary classification"""
    greater_is_better = True
    score_needs_proba = True
    name = "AUC"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.roc_auc_score(y_true, y_predicted)


class AUCMicro(MultiClassificationObjective):
    """AUC score for multiclass classification using micro averaging"""
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Micro"

    def objective_function(self, y_predicted, y_true, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='micro')


class AUCMacro(MultiClassificationObjective):
    """AUC score for multiclass classification using macro averaging"""
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Macro"

    def objective_function(self, y_predicted, y_true, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='macro')


class AUCWeighted(MultiClassificationObjective):
    """AUC Score for multiclass classification using weighted averaging"""
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Weighted"

    def objective_function(self, y_predicted, y_true, X=None):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='weighted')


class LogLossBinary(BinaryClassificationObjective):
    """Log Loss for binary classification"""
    greater_is_better = False
    score_needs_proba = True
    name = "Log Loss Binary"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.log_loss(y_true, y_predicted)


class LogLossMulticlass(MultiClassificationObjective):
    """Log Loss for multiclass classification"""
    greater_is_better = False
    score_needs_proba = True
    name = "Log Loss Multiclass"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.log_loss(y_true, y_predicted)


class MCCBinary(BinaryClassificationObjective):
    """Matthews correlation coefficient for binary classification"""
    greater_is_better = True
    name = "MCC Binary"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.matthews_corrcoef(y_true, y_predicted)


class MCCMulticlass(MultiClassificationObjective):
    """Matthews correlation coefficient for multiclass classification"""
    greater_is_better = True
    name = "MCC Multiclass"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.matthews_corrcoef(y_true, y_predicted)


class R2(RegressionObjective):
    """Coefficient of determination for regression"""
    greater_is_better = True
    name = "R2"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.r2_score(y_true, y_predicted)


class MAE(RegressionObjective):
    """Mean absolute error for regression"""
    greater_is_better = False
    name = "MAE"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.mean_absolute_error(y_true, y_predicted)


class MSE(RegressionObjective):
    """Mean squared error for regression"""
    greater_is_better = False
    name = "MSE"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.mean_squared_error(y_true, y_predicted)


class MSLE(RegressionObjective):
    """Mean squared log error for regression"""
    greater_is_better = False
    name = "MSLE"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.mean_squared_log_error(y_true, y_predicted)


class MedianAE(RegressionObjective):
    """Median absolute error for regression"""
    greater_is_better = False
    name = "MedianAE"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.median_absolute_error(y_true, y_predicted)


class MaxError(RegressionObjective):
    """Maximum residual error for regression"""
    greater_is_better = False
    name = "MaxError"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.max_error(y_true, y_predicted)


class ExpVariance(RegressionObjective):
    """Explained variance score for regression"""
    greater_is_better = True
    name = "ExpVariance"

    def objective_function(self, y_predicted, y_true, X=None):
        return metrics.explained_variance_score(y_true, y_predicted)


class PlotMetric(ABC):
    score_needs_proba = False
    name = None

    @abstractmethod
    def score(self, y_predicted, y_true):
        raise NotImplementedError("score() is not implemented!")


class ROC(PlotMetric):
    """Receiver Operating Characteristic score for binary classification."""
    score_needs_proba = True
    name = "ROC"

    def score(self, y_predicted, y_true):
        return metrics.roc_curve(y_true, y_predicted)


class ConfusionMatrix(PlotMetric):
    """Confusion matrix for classification problems"""
    name = "Confusion Matrix"

    def score(self, y_predicted, y_true):
        labels = unique_labels(y_predicted, y_true)
        conf_mat = metrics.confusion_matrix(y_true, y_predicted)
        conf_mat = pd.DataFrame(conf_mat, columns=labels)
        return conf_mat


def _handle_predictions(y_true, y_pred):
    if len(np.unique(y_true)) > 2:
        classes = np.unique(y_true)
        y_true = label_binarize(y_true, classes=classes)

    return y_true, y_pred
