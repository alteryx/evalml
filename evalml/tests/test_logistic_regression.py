import numpy as np

from evalml.objectives import PrecisionMicro
from evalml.pipelines import LogisticRegressionPipeline


def test_lr_multi(X_y_multi):
    X, y = X_y_multi
    objective = PrecisionMicro()
    clf = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', drop_invariant=False, number_features=len(X[0]))
    clf.fit(X, y)
    clf.score(X, y)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
