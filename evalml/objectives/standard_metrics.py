import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


# todo does this need tuning?
class F1(ObjectiveBase):
    """F1 score for binary classification"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "F1"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted)


class F1Micro(ObjectiveBase):
    """F1 score for multiclass classification using micro averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "F1 Micro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='micro')


class F1Macro(ObjectiveBase):
    """F1 score for multiclass classification using macro averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "F1 Macro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='macro')


class F1Weighted(ObjectiveBase):
    """F1 score for multiclass classification using weighted averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "F1 Weighted"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='weighted')


class Precision(ObjectiveBase):
    """Precision score for binary classification"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "Precision"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted)


class PrecisionMicro(ObjectiveBase):
    """Precision score for multiclass classification using micro averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "Precision Micro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='micro')


class PrecisionMacro(ObjectiveBase):
    """Precision score for multiclass classification using macro averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "Precision Macro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='macro')


class PrecisionWeighted(ObjectiveBase):
    """Precision score for multiclass classification using weighted averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "Precision Weighted"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='weighted')


class Recall(ObjectiveBase):
    """Recall score for binary classification"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "Recall"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.recall_score(y_true, y_predicted)


class RecallMicro(ObjectiveBase):
    """Recall score for multiclass classification using micro averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "Recall Micro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.recall_score(y_true, y_predicted, average='micro')


class RecallMacro(ObjectiveBase):
    """Recall score for multiclass classification using macro averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "Recall Macro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.recall_score(y_true, y_predicted, average='macro')


class RecallWeighted(ObjectiveBase):
    """Recall score for multiclass classification using weighted averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "Recall Weighted"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.recall_score(y_true, y_predicted, average='weighted')


class AUC(ObjectiveBase):
    """AUC score for binary classification"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = True
    name = "AUC"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.roc_auc_score(y_true, y_predicted)


class AUCMicro(ObjectiveBase):
    """AUC score for multiclass classification using micro averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Micro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='micro')


class AUCMacro(ObjectiveBase):
    """AUC score for multiclass classification using macro averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Macro"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='macro')


class AUCWeighted(ObjectiveBase):
    """AUC Score for multiclass classification using weighted averaging"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = True
    name = "AUC Weighted"
    problem_types = [ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='weighted')


class LogLoss(ObjectiveBase):
    """Log Loss for both binary and multiclass classification"""
    needs_fitting = False
    greater_is_better = False
    score_needs_proba = True
    name = "Log Loss"
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.log_loss(y_true, y_predicted)


class MCC(ObjectiveBase):
    """Matthews correlation coefficient for both binary and multiclass classification"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "MCC"
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def score(self, y_predicted, y_true):
        return metrics.matthews_corrcoef(y_true, y_predicted)


class R2(ObjectiveBase):
    """Coefficient of determination for regression"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "R2"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.r2_score(y_true, y_predicted)


class MAE(ObjectiveBase):
    """Mean absolute error for regression"""
    needs_fitting = False
    greater_is_better = False
    score_needs_proba = False
    name = "MAE"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.mean_absolute_error(y_true, y_predicted)


class MSE(ObjectiveBase):
    """Mean squared error for regression"""
    needs_fitting = False
    greater_is_better = False
    score_needs_proba = False
    name = "MSE"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.mean_squared_error(y_true, y_predicted)


class MSLE(ObjectiveBase):
    """Mean squared log error for regression"""
    needs_fitting = False
    greater_is_better = False
    score_needs_proba = False
    name = "MSLE"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.mean_squared_log_error(y_true, y_predicted)


class MedianAE(ObjectiveBase):
    """Median absolute error for regression"""
    needs_fitting = False
    greater_is_better = False
    score_needs_proba = False
    name = "MedianAE"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.median_absolute_error(y_true, y_predicted)


class MaxError(ObjectiveBase):
    """Maximum residual error for regression"""
    needs_fitting = False
    greater_is_better = False
    score_needs_proba = False
    name = "MaxError"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.max_error(y_true, y_predicted)


class ExpVariance(ObjectiveBase):
    """Explained variance score for regression"""
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = False
    name = "ExpVariance"
    problem_types = [ProblemTypes.REGRESSION]

    def score(self, y_predicted, y_true):
        return metrics.explained_variance_score(y_true, y_predicted)


class ROC(ObjectiveBase):
    """Receiver Operating Characteristic score for binary classification."""
    score_needs_proba = True
    name = "ROC"
    problem_types = [ProblemTypes.BINARY]

    def score(self, y_predicted, y_true):
        return metrics.roc_curve(y_true, y_predicted)


class ConfusionMatrix(ObjectiveBase):
    """Confusion matrix for classification problems"""
    needs_fitting = False
    greater_is_better = True
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
