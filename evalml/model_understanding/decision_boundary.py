"""Model Understanding for decision boundary on Binary Classification problems."""
import numpy as np
import pandas as pd

from evalml.pipelines import BinaryClassificationPipeline
from evalml.utils import infer_feature_types


# these are helper functions to help us calculate the objective values
def _accuracy(val_list):
    """Helper function to help us find the accuracy.

    The input expected should be [tp, tn, fp, fn]
    """
    acc = sum(val_list[:2]) / sum(val_list)
    return acc


def _balanced_accuracy(val_list):
    """Helper function to help us find the balanced accuracy.

    The input expected should be [tp, tn, fp, fn]
    """
    sens = _recall(val_list)
    if val_list[1] == 0:
        spec = 0
    else:
        spec = val_list[1] / (val_list[1] + val_list[2])
    return (sens + spec) / 2


def _precision(val_list):
    """Helper function to help us find the precision.

    The input expected should be [tp, tn, fp, fn]
    """
    if val_list[0] == 0:
        return 0
    return val_list[0] / (val_list[0] + val_list[2])


def _recall(val_list):
    """Helper function to help us find the recall.

    The input expected should be [tp, tn, fp, fn]
    """
    if val_list[0] == 0:
        return 0
    return val_list[0] / (val_list[0] + val_list[3])


def _f1(val_list):
    """Helper function to help us find the F1 score.

    The input expected should be [tp, tn, fp, fn]
    """
    prec = _precision(val_list)
    rec = _recall(val_list)
    if prec * rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)


def _find_confusion_matrix_objective_threshold(pos_skew, neg_skew, ranges):
    """Iterates through the arrays and determines the ideal objective thresholds and confusion matrix values for each threshold value.

    Arguments:
        pos_skew (list): The number of rows per bin value for the actual postive values.
        neg_skew (list): The number of rows per bin value for the actual negative values.
        ranges (list): The bin ranges, spanning from 0.0 to 1.0. The length of this list - 1 is equal to the number of bins.

    Returns:
        tuple: The first element is a list of confusion matrix values at each threshold bin, and the second element
            is a dictionary with the ideal objective thresholds and associated objective scores.
    """
    thresh_conf_matrix_list = []
    num_fn, num_tn = 0, 0
    total_pos, total_neg = sum(pos_skew), sum(neg_skew)
    # in the dict, [0, 0] corresponds to [objective_score, threshold value]
    objective_dict = {
        "accuracy": [[0, 0], _accuracy],
        "balanced_accuracy": [[0, 0], _balanced_accuracy],
        "precision": [[0, 0], _precision],
        "recall": [[0, 0], _recall],
        "f1": [[0, 0], _f1],
    }
    for i, thresh_val in enumerate(ranges[:-1]):
        num_fn += pos_skew[i]
        num_tn += neg_skew[i]
        num_tp = total_pos - num_fn
        num_fp = total_neg - num_tn
        # this is also the confusion matrix
        val_list = [num_tp, num_tn, num_fp, num_fn]
        thresh_conf_matrix_list.append(val_list)

        # let's iterate through the list to find the vals
        for k, v in objective_dict.items():
            obj_val = v[1](val_list)
            if obj_val > v[0][0]:
                v[0][0] = obj_val
                v[0][1] = thresh_val

        if num_fn == total_pos and num_tn == total_pos and i < len(ranges) - 1:
            # finished iterating through, there are no other changes
            v_extension = [val_list for _ in range(i + 1, len(ranges) - 1)]
            thresh_conf_matrix_list.extend(v_extension)
            break

    return (thresh_conf_matrix_list, objective_dict)


def _find_data_between_ranges(data, ranges, top_k):
    """Finds the rows of the data that fall between each range.

    Arguments:
        data (pd.Series): The predicted probability values for the postive class.
        ranges (list): The threshold ranges defining the bins. Should include 0 and 1 as the first and last value.
        top_k (int): The number of row indices per bin to include as samples.

    Returns:
        list(list): Each list corresponds to the row indices that fall in the range provided.
    """
    results = []
    for i in range(1, len(ranges)):
        mask = data[(data >= ranges[i - 1]) & (data < ranges[i])]
        if top_k != -1:
            results.append(mask.index.tolist()[: min(len(mask), top_k)])
        else:
            results.append(mask.index.tolist())
    return results


def find_confusion_matrix_per_thresholds(pipeline, X, y, n_bins=None, top_k=5):
    """Gets the confusion matrix and histogram bins for each threshold as well as the best threshold per objective. Only works with Binary Classification Pipelines.

    Arguments:
        pipeline (PipelineBase): A fitted Binary Classification Pipeline to get the confusion matrix with.
        X (pd.DataFrame): The input features.
        y (pd.Series): The input target.
        n_bins (int): The number of bins to use to calculate the threshold values. Defaults to None, which will default to using Freedman-Diaconis rule.
        top_k (int): The maximum number of row indices per bin to include as samples. -1 includes all row indices that fall between the bins. Defaults to 5.

    Returns:
        (pd.DataFrame, dict): The dataframe has the actual positive histogram, actual negative histogram,
            and the confusion matrix, all for each threshold value. The dictionary contains the ideal threshold and score per objective,
            keyed by objective name.

    Raises:
        ValueError: If the pipeline isn't a binary classification pipeline or isn't yet fitted on data.
    """
    if not pipeline._is_fitted or not isinstance(
        pipeline, BinaryClassificationPipeline
    ):
        raise ValueError("Expected a fitted binary classification pipeline")
    X = infer_feature_types(X)
    y = infer_feature_types(y)
    if set(y.unique()) != {0, 1}:
        y = pipeline._encode_targets(y)

    proba = pipeline.predict_proba(X)
    pos_preds = proba.iloc[:, -1]
    true_pos = y[y == 1]
    true_neg = y[y == 0]
    # separate the positive and negative predictions
    true_pos_preds = pos_preds.loc[true_pos.index]
    true_neg_preds = pos_preds.loc[true_neg.index]

    # get the histograms for the predictions
    if n_bins is not None:
        bins = [i / n_bins for i in range(n_bins + 1)]
    else:
        bins = np.histogram_bin_edges(pos_preds, bins="fd", range=(0, 1))
    pos_skew, pos_range = np.histogram(true_pos_preds, bins=bins)
    neg_skew, neg_range = np.histogram(true_neg_preds, bins=bins)
    data_ranges = _find_data_between_ranges(pos_preds, pos_range, top_k)

    conf_matrix_list, objective_dict = _find_confusion_matrix_objective_threshold(
        pos_skew, neg_skew, pos_range
    )
    result_df = pd.DataFrame(
        {
            "pos_bins": pos_skew,
            "neg_bins": neg_skew,
            "confusion_matrix": conf_matrix_list,
            "data_in_bins": data_ranges,
        },
        index=pos_range[:-1],
    )
    final_obj_dict = {k: v[0] for k, v in objective_dict.items()}
    return (result_df, final_obj_dict)
