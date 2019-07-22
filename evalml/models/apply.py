"""
Copyright (c) 2019 Feature Labs, Inc.

The usage of this software is governed by the Feature Labs End User License Agreement available at https://www.featurelabs.com/eula/. If you do not agree to the terms set out in this agreement, do not use the software, and immediately contact Feature Labs or your supplier.
"""
import pandas as pd
from dask import dataframe as dd
from numpy import unique
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.multiclass import type_of_target

from . import render


def load_data(path, index, label, drop=None, verbose=True, **kwargs):
    if '*' in path:
        feature_matrix = dd.read_csv(path, **kwargs).set_index(index, sorted=True)

        labels = [label] + (drop or [])
        y = feature_matrix[label].compute()
        X = feature_matrix.drop(labels=labels, axis=1).compute()
    else:
        feature_matrix = pd.read_csv(path, index_col=index, **kwargs)

        labels = [label] + (drop or [])
        y = feature_matrix[label]
        X = feature_matrix.drop(columns=labels)

    if verbose:
        # number of features
        print(render.number_of_features(X.dtypes), end='\n\n')

        # number of training examples
        info = 'Number of training examples: {}'
        print(info.format(len(X)), end='\n\n')

        # label distribution
        distribution = y.value_counts().div(len(y))
        print(render.label_distribution(distribution))

    return X, y


def split_data(x, y, holdout=.2, random_state=None):
    stratified = StratifiedShuffleSplit(
        n_splits=1,
        test_size=holdout,
        random_state=random_state,
    )
    train, test = next(stratified.split(x, y))
    x_train = x.loc[x.index[train]]
    x_test = x.loc[x.index[test]]
    y_train = y.loc[y.index[train]]
    y_test = y.loc[y.index[test]]
    return x_train, x_test, y_train, y_test


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
