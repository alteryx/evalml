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
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X[0]),
            "n_estimators": 20,
            "max_depth": 5
        },
        'Random Forest Classifier': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }

    clf = RFClassificationPipeline(objective=objective, parameters=parameters)

    expected_parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean',
            'fill_value': None
        },
        'One Hot Encoder': {'top_n': 10},
        'RF Classifier Select From Model': {
            'percent_features': 1.0,
            'threshold': -np.inf
        },
        'Random Forest Classifier': {
            'max_depth': 5,
            'n_estimators': 20
        }
    }

    assert clf.parameters == expected_parameters


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
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X[0]),
            "n_estimators": 10
        },
        'Random Forest Classifier': {
            "n_estimators": 10,
            "max_depth": 3
        }
    }
    clf = RFClassificationPipeline(objective=objective, parameters=parameters)
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
    parameters = {
        'Simple Imputer': {
            'impute_strategy': 'mean'
        },
        'RF Classifier Select From Model': {
            "percent_features": 1.0,
            "number_features": len(X.columns),
            "n_estimators": 20
        },
        'Random Forest Classifier': {
            "n_estimators": 20,
            "max_depth": 5,
        }
    }
    clf = RFClassificationPipeline(objective=objective, parameters=parameters)
    clf.fit(X, y)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
