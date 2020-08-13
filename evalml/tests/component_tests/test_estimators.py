import string

import numpy as np
import pandas as pd

from evalml.pipelines.components.utils import _all_estimators_used_in_search
from evalml.problem_types import ProblemTypes, handle_problem_types


def test_estimators_feature_name_with_random_ascii(X_y_binary, X_y_multi, X_y_regression):
    for estimator_class in _all_estimators_used_in_search():
        supported_problem_types = [handle_problem_types(pt) for pt in estimator_class.supported_problem_types]
        for problem_type in supported_problem_types:
            clf = estimator_class()
            if problem_type == ProblemTypes.BINARY:
                X, y = X_y_binary
            elif problem_type == ProblemTypes.MULTICLASS:
                X, y = X_y_multi
            elif problem_type == ProblemTypes.REGRESSION:
                X, y = X_y_regression

            X = clf.random_state.random((X.shape[0], len(string.printable)))
            col_names = ['column_{}'.format(ascii_char) for ascii_char in string.printable]
            X = pd.DataFrame(X, columns=col_names)
            clf.fit(X, y)
            assert len(clf.feature_importance) == len(X.columns)
            assert not np.isnan(clf.feature_importance).all().all()

            predictions = clf.predict(X)
            assert len(predictions) == len(y)
            assert not np.isnan(predictions).all()


def test_binary_classification_estimators_predict_proba_col_order():
    data = np.concatenate([np.zeros(100), np.ones(100)])
    X = pd.DataFrame({'input': data})
    y = pd.Series(data)
    for estimator_class in _all_estimators_used_in_search():
        supported_problem_types = [handle_problem_types(pt) for pt in estimator_class.supported_problem_types]
        if ProblemTypes.BINARY in supported_problem_types:
            estimator = estimator_class()
            estimator.fit(X, y)
            predicted_proba = estimator.predict_proba(X)
            expected = np.concatenate([(1 - data).reshape(-1, 1), data.reshape(-1, 1)], axis=1)
            np.testing.assert_allclose(expected, np.round(predicted_proba).values)
