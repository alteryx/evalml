import numpy as np

from evalml.objectives import PrecisionMicro
from evalml.pipelines import RFClassificationPipeline


def test_rf_multi(X_y_multi):
    X, y = X_y_multi
    objective = PrecisionMicro()
    clf = RFClassificationPipeline(objective=objective, n_estimators=10, max_depth=3, impute_strategy='mean', drop_invariant=False, percent_features=1.0, number_features=len(X[0]))
    clf.fit(X, y)
    clf.score(X, y)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) == 3
