import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import PrecisionMicro
from evalml.pipelines import RFClassificationPipeline


def test_rf_init(X_y):
    X, y = X_y

    objective = PrecisionMicro()
    clf = RFClassificationPipeline(objective=objective, n_estimators=20, max_depth=5, impute_strategy='mean', percent_features=1.0, number_features=len(X[0]), random_state=2)
    expected_parameters = {'impute_strategy': 'mean', 'percent_features': 1.0,
                           'threshold': -np.inf, 'n_estimators': 20, 'max_depth': 5}
    assert clf.parameters == expected_parameters
    assert clf.random_state == 2


def test_rf_multi(X_y_multi):
    X, y = X_y_multi

    # create sklearn pipeline
    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = RandomForestClassifier(random_state=0,
                                       n_estimators=10,
                                       max_depth=3,
                                       n_jobs=-1)
    feature_selection = SelectFromModel(estimator=estimator,
                                        max_features=max(1, int(1 * len(X[0]))),
                                        threshold=-np.inf)
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("feature_selection", feature_selection),
                            ("estimator", estimator)])
    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)

    objective = PrecisionMicro()
    clf = RFClassificationPipeline(objective=objective, n_estimators=10, max_depth=3, impute_strategy='mean', percent_features=1.0, number_features=len(X[0]))
    clf.fit(X, y)
    clf_score = clf.score(X, y)
    y_pred = clf.predict(X)

    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_score[0])
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
    assert not clf.feature_importances.isnull().all().all()


def test_rf_input_feature_names(X_y):
    X, y = X_y
    # create a list of column names
    col_names = ["col_{}".format(i) for i in range(len(X[0]))]
    X = pd.DataFrame(X, columns=col_names)
    objective = PrecisionMicro()
    clf = RFClassificationPipeline(objective=objective, n_estimators=10, max_depth=3, impute_strategy='mean', percent_features=1.0, number_features=len(X.columns))
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    assert ("col_" in col_name for col_name in clf.feature_importances["feature"])
