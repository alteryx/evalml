import numpy as np
import pandas as pd

from evalml.objectives import PrecisionMicro
from evalml.pipelines import LogisticRegressionPipeline


def test_lr_multi(X_y_multi):
    X, y = X_y_multi
    X = pd.DataFrame(X)
    y = pd.Series(y)

    objective = PrecisionMicro()
    clf = LogisticRegressionPipeline(objective=objective, penalty='l2', C=1.0, impute_strategy='mean', number_features=0)
    clf.fit(X, y)
    clf.score(X, y)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) == 3
