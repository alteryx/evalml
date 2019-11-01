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
