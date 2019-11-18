from collections import OrderedDict

import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold

from evalml import PipelineSearchPlots
from evalml.models.auto_base import AutoBase
from evalml.pipelines import LogisticRegressionPipeline


def test_generate_roc(X_y):
    X, y = X_y

    # Make mock class and generate mock results

    class MockAuto(AutoBase):
        def __init__(self):

            pipeline = LogisticRegressionPipeline(objective="ROC", penalty='l2', C=0.5,
                                                  impute_strategy='mean', number_features=len(X[0]), random_state=1)
            cv = StratifiedKFold(n_splits=5, random_state=0)
            results = {}
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

            results.update({"results": {0: {"cv_data": cv_data,
                                            "pipeline_name": pipeline.name}}})
            self.results = results

    mock_clf = MockAuto()

    search_plots = PipelineSearchPlots(mock_clf.results)

    roc_data = search_plots.get_roc_data(0)
    assert len(roc_data["fpr_tpr_data"]) == 5
    assert len(roc_data["roc_aucs"]) == 5

    fig = search_plots.generate_roc_plot(0)
    assert isinstance(fig, type(go.FigureWidget()))
