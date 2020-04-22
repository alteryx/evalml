import warnings

import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.utils.multiclass import unique_labels


def roc_curve(y_true, y_predicted):
    """Receiver Operating Characteristic score for binary classification."""
    return roc_curve(y_true, y_predicted)


def confusion_matrix(y_true, y_predicted):
    """Confusion matrix for binary and multiclass classification problems"""
    labels = unique_labels(y_predicted, y_true)
    conf_mat = confusion_matrix(y_true, y_predicted)
    conf_mat = pd.DataFrame(conf_mat, columns=labels)
    return conf_mat


def normalize_confusion_matrix(conf_mat, option='true'):
    """Normalizes a confusion matrix.

    Arguments:
        conf_mat (pd.DataFrame or np.array): confusion matrix to normalize
        option ({'true', 'pred', 'all', None}): Option to normalize over the rows ('true'), columns ('pred') or all ('all') values. If option is None, returns original confusion matrix. Defaults to 'true'.

    Returns:
        A normalized version of the input confusion matrix.

    """
    with warnings.catch_warnings(record=True) as w:
        if option is None:
            return conf_mat
        elif option == 'true':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        elif option == 'pred':
            conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=0)
        elif option == 'all':
            conf_mat = conf_mat.astype('float') / conf_mat.sum().sum()

        if w and "invalid value encountered in" in str(w[0].message):
            raise ValueError("Sum of given axis is 0 and normalization is not possible. Please select another option.")

    return conf_mat
