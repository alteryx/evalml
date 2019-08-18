"""
Copyright (c) 2019 Feature Labs, Inc.

The usage of this software is governed by the Feature Labs End User License Agreement available at https://www.featurelabs.com/eula/. If you do not agree to the terms set out in this agreement, do not use the software, and immediately contact Feature Labs or your supplier.
"""
import pandas as pd
from numpy import unique
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target

from . import render


def resample_labels(x, y, weights, replace=False, label=None, random_state=None):
    if label is None:
        length = len(y)
    else:
        length = y.value_counts().loc[label]

    concat_x, concat_y = [], []
    for label, weight, in weights.items():
        index, n = y.eq(label), int(weight * length)
        concat_x.append(x.loc[index].sample(n=n, replace=replace, random_state=random_state))
        concat_y.append(y.loc[index].sample(n=n, replace=replace, random_state=random_state))
    x, y = pd.concat(concat_x, axis=0), pd.concat(concat_y, axis=0)
    return x, y


def is_binary(labels):
    return type_of_target(labels) == 'binary' and unique(labels).size == 2


def score(y, y_hat):
    scores = {}
    binary = is_binary(y) and is_binary(y_hat)
    if binary:
        scores.update({
            'F1': metrics.f1_score(y, y_hat),
            'Precision': metrics.precision_score(y, y_hat),
            'Recall': metrics.recall_score(y, y_hat),
            'Accuracy': metrics.accuracy_score(y, y_hat),
            'AUC': metrics.roc_auc_score(y, y_hat),
            'Log Loss': metrics.log_loss(y, y_hat),
            'MCC': metrics.matthews_corrcoef(y, y_hat),
        })
    return scores


def compare_scores(test, cv):
    # calcualte whether scores are within estimated range
    mean, sd = cv['Mean'], cv['Standard Deviation']
    ge_lower_bound = test.Scores.ge(mean.sub(sd))
    le_upper_bound = test.Scores.le(mean.add(sd))
    test['Within Estimated Range'] = ge_lower_bound & le_upper_bound

    # calcualte the normalized difference
    difference = test.Scores.sub(cv.Mean)
    percent = difference.div(cv.Mean).mul(100)
    test['Diff from Mean Estimate'] = percent.apply('{:+.2f}%'.format)

    return test
