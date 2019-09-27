# import numpy as np
# import pandas as pd

from evalml.pipelines import (Estimator,  
                              LogisticRegressionClassifier,
                              RandomForestClassifier,
                              XGBoostClassifier,
                              RandomForestRegressor,
                              LinearRegressor)

# Tests to include:
#   for each specific estimator
#   for an user-defined estimator 

def test_init(X_y):
    lr_classifier = LogisticRegressionClassifier()
    rf_classifier = RandomForestClassifier(n_estimators=10)
    xgb_classifier = XGBoostClassifier(eta=0.1, min_child_weight=1, max_depth=3)
    rf_regressor = RandomForestRegressor(n_estimators=10)
    linear_regressor = LinearRegressor()
    assert lr_classifier.component_type == 'classifier'
    assert rf_classifier.component_type == 'classifier'
    assert xgb_classifier.component_type == 'classifier'
    assert rf_regressor.component_type == 'regressor'
    assert linear_regressor.component_type == 'regressor'
