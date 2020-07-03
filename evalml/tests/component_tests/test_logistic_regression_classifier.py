import string

import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import LogisticRegressionClassifier


@pytest.mark.parametrize('problem_type', ['binary', 'multiclass'])
def test_logistic_regression_feature_name_with_random_ascii(problem_type, X_y_binary, X_y_multi):
    if problem_type == 'binary':
        X, y = X_y_binary
    else:
        X, y = X_y_multi
    clf = LogisticRegressionClassifier()
    X = clf.random_state.random((X.shape[0], len(string.printable)))
    col_names = ['column_{}'.format(ascii_char) for ascii_char in string.printable]
    X = pd.DataFrame(X, columns=col_names)
    clf.fit(X, y)
    assert len(clf.feature_importance) == len(X.columns)
    assert not np.isnan(clf.feature_importance).all().all()

    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert not np.isnan(predictions).all()
