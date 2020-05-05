import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import roc_curve as sklearn_roc_curve
from sklearn.utils.multiclass import unique_labels


def roc_curve(y_true, y_pred_proba):
    """Receiver Operating Characteristic score for binary classification.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_pred_proba (pd.Series or np.array): predictions from a binary classifier, before thresholding has been applied.

    Returns:
        (np.array, np.array, np.array): false positive rates, true positive rates, and threshold values used to produce each pair of true/false positive rates.
    """
    return sklearn_roc_curve(y_true, y_pred_proba)


def confusion_matrix(y_true, y_predicted):
    """Confusion matrix for binary and multiclass classification.

    Arguments:
        y_true (pd.Series or np.array): true binary labels.
        y_predicted (pd.Series or np.array): predictions from a binary classifier, before thresholding has been applied.

    Returns:
        np.array: confusion matrix
    """
    labels = unique_labels(y_true, y_predicted)
    conf_mat = sklearn_confusion_matrix(y_true, y_predicted)
    conf_mat = pd.DataFrame(conf_mat, columns=labels)
    return conf_mat


def normalize_confusion_matrix(conf_mat, option='true'):
    """Normalizes a confusion matrix.

    Arguments:
        conf_mat (pd.DataFrame or np.array): confusion matrix to normalize
        option ({'true', 'pred', 'all'}): Option to normalize over the rows ('true'), columns ('pred') or all ('all') values. Defaults to 'true'.

    Returns:
        A normalized version of the input confusion matrix.
    """
    with warnings.catch_warnings(record=True) as w:
        if option == 'true':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        elif option == 'pred':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=0)
        elif option == 'all':
            conf_mat = conf_mat.astype('float') / conf_mat.sum().sum()
        else:
            raise ValueError('Invalid value provided for "option": %s'.format(option))

        if w and "invalid value encountered in" in str(w[0].message):
            raise ValueError("Sum of given axis is 0 and normalization is not possible. Please select another option.")

    return conf_mat
