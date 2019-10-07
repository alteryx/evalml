import category_encoders as ce
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from evalml.objectives import PrecisionMicro
from evalml.pipelines import XGBoostPipeline


def test_xg_multi(X_y_multi):
    X, y = X_y_multi
    objective = PrecisionMicro()
    clf = XGBoostPipeline(objective=objective, eta=0.1, min_child_weight=1, max_depth=3, impute_strategy='mean', percent_features=1.0, number_features=len(X[0]))
    clf.fit(X, y)
    clf_score = clf.score(X, y)
    y_pred = clf.predict(X)

    imputer = SimpleImputer(strategy='mean')
    enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)
    estimator = XGBClassifier(random_state=0,
                              eta=0.1,
                              max_depth=3,
                              min_child_weight=1)
    feature_selection = SelectFromModel(
        estimator=estimator,
        max_features=max(1, int(1 * len(X[0]))),
        threshold=-np.inf
    )
    sk_pipeline = Pipeline([("encoder", enc),
                            ("imputer", imputer),
                            ("feature_selection", feature_selection),
                            ("estimator", estimator)])

    sk_pipeline.fit(X, y)
    sk_score = sk_pipeline.score(X, y)
    assert((y_pred == sk_pipeline.predict(X)).all())
    assert (sk_score == clf_score[0])
    assert len(np.unique(y_pred)) == 3
