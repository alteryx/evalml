from evalml.objectives import Precision
from evalml.pipelines import RFClassificationPipeline

import numpy as np

def test_rf_multi(X_y_multi):
    X, y = X_y_multi
    objective = Precision(average='micro')
    clf = RFClassificationPipeline(objective=objective, n_estimators=10, max_depth=3, impute_strategy='mean', percent_features=1.0, number_features=0)
    clf.fit(X, y)
    clf.score(X, y)
    y_pred = clf.predict(X)
    assert len(np.unique(y_pred)) == 3