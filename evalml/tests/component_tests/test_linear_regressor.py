import string

import numpy as np
import pandas as pd

from evalml.pipelines.components import LinearRegressor


def test_linear_regressor_feature_name_with_random_ascii(X_y_regression):
    X, y = X_y_regression
    clf = LinearRegressor()
    X = clf.random_state.random((X.shape[0], len(string.printable)))
    col_names = ['column_{}'.format(ascii_char) for ascii_char in string.printable]
    X = pd.DataFrame(X, columns=col_names)
    clf.fit(X, y)
    assert len(clf.feature_importance) == len(X.columns)
    assert not np.isnan(clf.feature_importance).all().all()

    predictions = clf.predict(X)
    assert len(predictions) == len(y)
    assert not np.isnan(predictions).all()
