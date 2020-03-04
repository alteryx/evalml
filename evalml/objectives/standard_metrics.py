from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

from .binary_classification_objective import BinaryClassificationObjective
from .multiclass_classification_objective import MultiClassificationObjective
from .objective_base import ObjectiveBase
from .regression_objective import RegressionObjective

from evalml.problem_types import ProblemTypes


# todo does this need tuning?
class F1(BinaryClassificationObjective):
    """F1 score for binary classification"""
    greater_is_better = True
    score_needs_proba = False
    name = "F1"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted)


class F1Micro(MultiClassificationObjective):
    """F1 score for multiclass classification using micro averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "F1 Micro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='micro')


class F1Macro(MultiClassificationObjective):
    """F1 score for multiclass classification using macro averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "F1 Macro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='macro')


class F1Weighted(MultiClassificationObjective):
    """F1 score for multiclass classification using weighted averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "F1 Weighted"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='weighted')


class Precision(BinaryClassificationObjective):
    """Precision score for binary classification"""
    greater_is_better = True
    score_needs_proba = False
    name = "Precision"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted)


class PrecisionMicro(MultiClassificationObjective):
    """Precision score for multiclass classification using micro averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "Precision Micro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='micro')


class PrecisionMacro(MultiClassificationObjective):
    """Precision score for multiclass classification using macro averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "Precision Macro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='macro')


class PrecisionWeighted(MultiClassificationObjective):
    """Precision score for multiclass classification using weighted averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "Precision Weighted"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='weighted')


class Recall(BinaryClassificationObjective):
    """Recall score for binary classification"""
    greater_is_better = True
    score_needs_proba = False
    name = "Recall"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.recall_score(y_true, y_predicted)


class RecallMicro(MultiClassificationObjective):
    """Recall score for multiclass classification using micro averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "Recall Micro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.recall_score(y_true, y_predicted, average='micro')


class RecallMacro(MultiClassificationObjective):
    """Recall score for multiclass classification using macro averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "Recall Macro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.recall_score(y_true, y_predicted, average='macro')


class RecallWeighted(MultiClassificationObjective):
    """Recall score for multiclass classification using weighted averaging"""
    greater_is_better = True
    score_needs_proba = False
    name = "Recall Weighted"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.recall_score(y_true, y_predicted, average='weighted')


class AUC(BinaryClassificationObjective):
    """AUC score for binary classification"""
    greater_is_better = True
    score_needs_proba = True
    name = "AUC"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.roc_auc_score(y_true, y_predicted)


class AUCMicro(MultiClassificationObjective):
    """AUC score for multiclass classification using micro averaging"""
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Micro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='micro')


class AUCMacro(MultiClassificationObjective):
    """AUC score for multiclass classification using macro averaging"""
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Macro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='macro')


class AUCWeighted(MultiClassificationObjective):
    """AUC Score for multiclass classification using weighted averaging"""
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Weighted"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='weighted')


class LogLoss(ObjectiveBase):
    """Log Loss for both binary and multiclass classification"""
    greater_is_better = False
    score_needs_proba = True
    name = "Log Loss"
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.log_loss(y_true, y_predicted)


class MCC(ObjectiveBase):
    """Matthews correlation coefficient for both binary and multiclass classification"""
    greater_is_better = True
    score_needs_proba = False
    name = "MCC"
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.matthews_corrcoef(y_true, y_predicted)


class R2(RegressionObjective):
    """Coefficient of determination for regression"""
    greater_is_better = True
    score_needs_proba = False
    name = "R2"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.r2_score(y_true, y_predicted)


class MAE(RegressionObjective):
    """Mean absolute error for regression"""
    greater_is_better = False
    score_needs_proba = False
    name = "MAE"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.mean_absolute_error(y_true, y_predicted)


class MSE(RegressionObjective):
    """Mean squared error for regression"""
    greater_is_better = False
    score_needs_proba = False
    name = "MSE"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.mean_squared_error(y_true, y_predicted)


class MSLE(RegressionObjective):
    """Mean squared log error for regression"""
    greater_is_better = False
    score_needs_proba = False
    name = "MSLE"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.mean_squared_log_error(y_true, y_predicted)


class MedianAE(RegressionObjective):
    """Median absolute error for regression"""
    greater_is_better = False
    score_needs_proba = False
    name = "MedianAE"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.median_absolute_error(y_true, y_predicted)


class MaxError(RegressionObjective):
    """Maximum residual error for regression"""
    greater_is_better = False
    score_needs_proba = False
    name = "MaxError"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.max_error(y_true, y_predicted)


class ExpVariance(RegressionObjective):
    """Explained variance score for regression"""
    greater_is_better = True
    score_needs_proba = False
    name = "ExpVariance"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.explained_variance_score(y_true, y_predicted)


class PlotMetric(ABC):
    score_needs_proba = True
    name = None

    @abstractmethod
    def score(self, y_predicted, y_true):
        raise NotImplementedError("score() is not implemented!")


class ROC(PlotMetric):
    """Receiver Operating Characteristic score for binary classification."""
    score_needs_proba = True
    name = "ROC"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.roc_curve(y_true, y_predicted)


class ConfusionMatrix(PlotMetric):
    """Confusion matrix for classification problems"""
    score_needs_proba = False
    name = "Confusion Matrix"
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

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
