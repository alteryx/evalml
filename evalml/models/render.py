"""
Copyright (c) 2019 Feature Labs, Inc.

The usage of this software is governed by the Feature Labs End User License Agreement available at https://www.featurelabs.com/eula/. If you do not agree to the terms set out in this agreement, do not use the software, and immediately contact Feature Labs or your supplier.
"""
from pandas import DataFrame, Series


def feature_importances(scores, **kwargs):
    scores = Series(scores, name='Score').rename_axis('Features')
    return scores.sort_values(ascending=False)


def feature_importances_plot(feature_importances, **kwargs):
    kwargs['figsize'] = kwargs.get('figsize', (10, 10))
    plot = feature_importances.sort_values().plot(kind='barh', **kwargs)
    plot.set_title('Feature Importance')
    plot.set_xlabel('Score')


def scores_estimate(scores, title='Cross Validation'):
    columns = dict(mean='Mean', std='Standard Deviation')
    scores = DataFrame(scores).describe().T.rename(columns=columns)[list(columns.values())]
    return scores.rename_axis('Metrics').rename_axis(title, axis=1).round(4)


def scores(scores):
    df = Series(scores).round(4).to_frame(name='Scores').sort_index()
    return df.rename_axis('Metrics')


def label_distribution(labels):
    return labels.mul(100).apply('{:.2f}%'.format).rename_axis('Labels')


def number_of_features(dtypes):
    dtype_to_vtype = {
        'bool': 'Boolean',
        'int32': 'Numeric',
        'int64': 'Numeric',
        'float64': 'Numeric',
        'object': 'Categorical',
        'datetime64[ns]': 'Datetime',
    }

    vtypes = dtypes.astype(str).map(dtype_to_vtype).value_counts()
    return vtypes.sort_index().to_frame('Number of Features')
