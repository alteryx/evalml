import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from evalml.objectives import PrecisionMicro
from evalml.pipelines import (
    RFBinaryClassificationPipeline,
    RFMulticlassClassificationPipeline
)


def test_rf_init(X_y):
    X, y = X_y

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

    clf = RFClassificationPipeline(parameters=parameters, random_state=2)

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
    assert (clf.random_state.get_state()[0] == np.random.RandomState(2).get_state()[0])


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
    clf = RFMulticlassClassificationPipeline(parameters=parameters)
    clf.fit(X, y)
    clf_scores = clf.score(X, y, [objective])
    y_pred = clf.predict(X)

    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_scores[objective.name])
    assert len(np.unique(y_pred)) == 3
    assert len(clf.feature_importances) == len(X[0])
    assert not clf.feature_importances.isnull().all().all()

    # testing objective parameter passed in does not change results
    clf.fit(X, y, objective)
    y_pred_with_objective = clf.predict(X)
    assert((y_pred == y_pred_with_objective).all())


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
    clf = RFBinaryClassificationPipeline(parameters=parameters)
    clf.fit(X, y, objective)
    assert len(clf.feature_importances) == len(X.columns)
    assert not clf.feature_importances.isnull().all().all()
    for col_name in clf.feature_importances["feature"]:
        assert "col_" in col_name
