from collections import OrderedDict

import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.model_selection import StratifiedKFold

from evalml.models.auto_base import AutoBase
from evalml.models.pipeline_search_plots import PipelineSearchPlots
from evalml.pipelines import LogisticRegressionPipeline
from evalml.problem_types import ProblemTypes


def test_generate_roc(X_y):
    X, y = X_y

    # Make mock class and generate mock results

    class MockAuto(AutoBase):
        def __init__(self):
            self.results = {}
            self.problem_type = ProblemTypes.BINARY

        def fit(self):
            pipeline = LogisticRegressionPipeline(objective="ROC", penalty='l2', C=0.5,
                                                  impute_strategy='mean', number_features=len(X[0]), random_state=1)
            cv = StratifiedKFold(n_splits=5, random_state=0)
            cv_data = []
            for train, test in cv.split(X, y):
                if isinstance(X, pd.DataFrame):
                    X_train, X_test = X.iloc[train], X.iloc[test]
                else:
                    X_train, X_test = X[train], X[test]
                if isinstance(y, pd.Series):
                    y_train, y_test = y.iloc[train], y.iloc[test]
                else:
                    y_train, y_test = y[train], y[test]

                pipeline.fit(X_train, y_train)
                score, other_scores = pipeline.score(X_test, y_test)

                ordered_scores = OrderedDict()
                ordered_scores.update({"ROC": score})
                ordered_scores.update({"# Training": len(y_train)})
                ordered_scores.update({"# Testing": len(y_test)})
                cv_data.append({"all_objective_scores": ordered_scores, "score": score})

            self.results.update({0: {"cv_data": cv_data,
                                     "pipeline_name": pipeline.name}})

    mock_clf = MockAuto()
    search_plots = PipelineSearchPlots(mock_clf)
    with pytest.raises(RuntimeError, match="You must first call fit"):
        search_plots.get_roc_data(0)
    with pytest.raises(RuntimeError, match="You must first call fit"):
        search_plots.generate_roc_plot(0)

    mock_clf.fit()

    with pytest.raises(RuntimeError, match="Pipeline 1 not found"):
        search_plots.get_roc_data(1)
    with pytest.raises(RuntimeError, match="Pipeline 1 not found"):
        search_plots.generate_roc_plot(1)

    roc_data = search_plots.get_roc_data(0)
    assert len(roc_data["fpr_tpr_data"]) == 5
    assert len(roc_data["roc_aucs"]) == 5

    fig = search_plots.generate_roc_plot(0)
    assert isinstance(fig, type(go.Figure()))


def test_generate_roc_multi_raises_errors(X_y):

    class MockAutoMulti(AutoBase):
        def __init__(self):
            self.results = {}
            self.problem_type = ProblemTypes.MULTICLASS

    mock_clf = MockAutoMulti()
    search_plots = PipelineSearchPlots(mock_clf)

    with pytest.raises(RuntimeError, match="ROC plots can only be generated for binary classification problems."):
        search_plots.get_roc_data(0)
    with pytest.raises(RuntimeError, match="ROC plots can only be generated for binary classification problems."):
        search_plots.generate_roc_plot(0)


def test_generate_confusion_matrix(X_y):
    X, y = X_y
    y_test_lens = []

    # Make mock class and generate mock results
    class MockAutoClassifier(AutoBase):
        def __init__(self):
            self.results = {}
            self.problem_type = ProblemTypes.BINARY

        def fit(self):
            pipeline = LogisticRegressionPipeline(objective="confusion_matrix", penalty='l2', C=0.5,
                                                  impute_strategy='mean', number_features=len(X[0]), random_state=1)
            cv = StratifiedKFold(n_splits=5, random_state=0)
            cv_data = []
            for train, test in cv.split(X, y):
                if isinstance(X, pd.DataFrame):
                    X_train, X_test = X.iloc[train], X.iloc[test]
                else:
                    X_train, X_test = X[train], X[test]
                if isinstance(y, pd.Series):
                    y_train, y_test = y.iloc[train], y.iloc[test]
                else:
                    y_train, y_test = y[train], y[test]

                # store information for testing purposes later
                y_test_lens.append(len(y_test))

                pipeline.fit(X_train, y_train)
                score, other_scores = pipeline.score(X_test, y_test)

                ordered_scores = OrderedDict()
                ordered_scores.update({"Confusion Matrix": score})
                ordered_scores.update({"# Training": len(y_train)})
                ordered_scores.update({"# Testing": len(y_test)})
                cv_data.append({"all_objective_scores": ordered_scores, "score": score})

            self.results.update({0: {"cv_data": cv_data,
                                     "pipeline_name": pipeline.name}})

    mock_clf = MockAutoClassifier()
    search_plots = PipelineSearchPlots(mock_clf)
    with pytest.raises(RuntimeError, match="You must first call fit"):
        search_plots.get_confusion_matrix_data(0)
    with pytest.raises(RuntimeError, match="You must first call fit"):
        search_plots.generate_confusion_matrix(0)

    mock_clf.fit()

    with pytest.raises(RuntimeError, match="Pipeline 1 not found"):
        search_plots.get_confusion_matrix_data(1)
    with pytest.raises(RuntimeError, match="Pipeline 1 not found"):
        search_plots.generate_confusion_matrix(1)

    cm_data = search_plots.get_confusion_matrix_data(0)
    for i, cm in enumerate(cm_data):
        labels = cm.columns
        assert all(label in y for label in labels)
        assert (cm.to_numpy().sum() == y_test_lens[i])
    fig = search_plots.generate_confusion_matrix(0)
    assert isinstance(fig, type(go.Figure()))


def test_confusion_matrix_regression_throws_error():

    # Make mock class and generate mock results
    class MockAutoRegressor(AutoBase):
        def __init__(self):
            self.results = {}
            self.problem_type = ProblemTypes.REGRESSION

    mock_clf = MockAutoRegressor()
    search_plots = PipelineSearchPlots(mock_clf)

    with pytest.raises(RuntimeError, match="Confusion matrix plots can only be generated for classification problems"):
        search_plots.get_confusion_matrix_data(0)
    with pytest.raises(RuntimeError, match="Confusion matrix plots can only be generated for classification problems."):
        search_plots.generate_confusion_matrix(0)
