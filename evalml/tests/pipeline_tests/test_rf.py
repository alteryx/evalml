import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import category_encoders as ce
from evalml.objectives import PrecisionMicro
from evalml.pipelines import RFClassificationPipeline


def test_rf_multi(X_y_multi):
    X, y = X_y_multi

    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = RandomForestClassifier(random_state=0,
                                       n_estimators=10,
                                       max_depth=3,
                                       n_jobs=-1)

    feature_selection = SelectFromModel(
        estimator=estimator,
        max_features=max(1, int(1 * len(X[0]))),
        threshold=-np.inf
    )

    sk_pipeline = Pipeline(
            [("encoder", enc),
            ("imputer", imputer),
            ("feature_selection", feature_selection),
            ("estimator", estimator)]
       )
    
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
