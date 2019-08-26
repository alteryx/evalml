import numpy as np

from evalml.objectives import Precision
from evalml.pipelines import LogisticRegressionPipeline


def test_lr_multi(X_y_multi):
    X, y = X_y_multi
    objective = Precision(average='micro')
    clf = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=0)
    clf.fit(X, y)
    clf.score(X, y)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) == 3
