
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold

from evalml import AutoClassifier

def test_pipeline_limits(capsys, X_y):
    X, y = X_y

    clf = AutoClassifier(multiclass=False, max_pipelines=1)
    clf.fit(X, y)
    out, err = capsys.readouterr()
    assert "Searching up to 1 pipelines. " in out

    clf = AutoClassifier(multiclass=False, max_time=1)
    clf.fit(X, y)
    out, err = capsys.readouterr()
    assert "Will stop searching for new pipelines after 1 seconds" in out

    clf = AutoClassifier(multiclass=False, max_time=1, max_pipelines=5)
    clf.fit(X, y)
    out, err = capsys.readouterr()
    assert "Searching up to 5 pipelines. " in out
    assert "Will stop searching for new pipelines after 1 seconds" in out

    clf = AutoClassifier(multiclass=False)
    clf.fit(X, y)
    out, err = capsys.readouterr()
    assert "No search limit is set. Set using max_time or max_pipelines." in out


def test_generate_roc(X_y):
    X, y = X_y
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, random_state=0)
    clf = AutoClassifier(multiclass=False, cv=cv, max_pipelines=2, random_state=0)
    clf.fit(X, y)
    roc_data = clf.plot.get_roc_data(0)
    assert len(roc_data["fpr_tpr_data"]) == 5
    assert len(roc_data["roc_aucs"]) == 5

    fig = clf.plot.generate_roc_plot(0)
    assert isinstance(fig, type(go.FigureWidget()))
