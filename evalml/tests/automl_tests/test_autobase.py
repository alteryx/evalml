import pytest
from sklearn.model_selection import StratifiedKFold

from evalml import AutoClassificationSearch
from evalml.pipelines import LogisticRegressionPipeline


def test_pipeline_limits(capsys, X_y):
    X, y = X_y

    automl = AutoClassificationSearch(multiclass=False, max_pipelines=1)
    automl.search(X, y, raise_errors=True)
    out, err = capsys.readouterr()
    assert "Searching up to 1 pipelines. " in out

    automl = AutoClassificationSearch(multiclass=False, max_time=1)
    automl.search(X, y, raise_errors=True)
    out, err = capsys.readouterr()
    assert "Will stop searching for new pipelines after 1 seconds" in out

    automl = AutoClassificationSearch(multiclass=False, max_time=1, max_pipelines=5)
    automl.search(X, y, raise_errors=True)
    out, err = capsys.readouterr()
    assert "Searching up to 5 pipelines. " in out
    assert "Will stop searching for new pipelines after 1 seconds" in out

    automl = AutoClassificationSearch(multiclass=False)
    automl.search(X, y, raise_errors=True)
    out, err = capsys.readouterr()
    assert "No search limit is set. Set using max_time or max_pipelines." in out


def test_generate_roc(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, random_state=0)
    automl = AutoClassificationSearch(multiclass=False, cv=cv, max_pipelines=2, random_state=0)
    automl.search(X, y, raise_errors=True)
    roc_data = automl.plot.get_roc_data(0)
    assert len(roc_data["fpr_tpr_data"]) == 5
    assert len(roc_data["roc_aucs"]) == 5

    fig = automl.plot.generate_roc_plot(0)
    assert isinstance(fig, type(go.Figure()))


def test_generate_confusion_matrix(X_y):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = X_y
    n_splits = 5
    cv = StratifiedKFold(n_splits=n_splits, random_state=0)
    automl = AutoClassificationSearch(multiclass=False, cv=cv, max_pipelines=2, random_state=0)
    automl.search(X, y, raise_errors=True)
    cm_data = automl.plot.get_confusion_matrix_data(0)
    assert len(cm_data) == 5
    for fold in cm_data:
        labels = fold.columns
        assert all(label in y for label in labels)

    fig = automl.plot.generate_confusion_matrix(0)
    assert isinstance(fig, type(go.Figure()))


def test_search_order(X_y):
    X, y = X_y
    automl = AutoClassificationSearch(max_pipelines=3)
    automl.search(X, y)
    correct_order = [0, 1, 2]
    assert automl.results['search_order'] == correct_order


def test_transform_parameters():
    automl = AutoClassificationSearch(max_pipelines=1, random_state=100, n_jobs=6)
    parameters = [('penalty', 'l2'), ('C', 8.444214828324364), ('impute_strategy', 'most_frequent')]
    parameters_dict = {
        'Simple Imputer': {'impute_strategy': 'most_frequent'},
        'One Hot Encoder': {'random_state': 100},
        'Standard Scaler': {},
        'Logistic Regression Classifier': {'penalty': 'l2', 'C': 8.444214828324364, 'n_jobs': 6}
    }
    assert automl._transform_parameters(LogisticRegressionPipeline, parameters, 0) == parameters_dict
