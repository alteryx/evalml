import numpy as np

from evalml.objectives import ROC, ConfusionMatrix


def test_confusion_matrix():
    y_true = [2, 0, 2, 2, 0, 1]
    y_predicted = [0, 0, 2, 2, 0, 2]
    cm = ConfusionMatrix()
    score = cm.score(y_predicted, y_true)
    cm_expected = np.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    assert np.array_equal(cm_expected, score)


def test_roc():
    y_true = np.array([1, 1, 0, 0])
    y_predict_proba = np.array([0.1, 0.4, 0.35, 0.8])
    roc_metric = ROC()
    fpr, tpr, thresholds = roc_metric.score(y_predict_proba, y_true)
    assert not np.isnan(fpr).any()
    assert not np.isnan(tpr).any()
    assert not np.isnan(thresholds).any()
