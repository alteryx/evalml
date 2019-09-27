import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skopt.space import Integer, Real

from evalml.pipelines import PipelineBase
from evalml.pipelines.components import RandomForestClassifier
from evalml.problem_types import ProblemTypes


class RFClassificationPipeline(PipelineBase):
    """Random Forest Pipeline for both binary and multiclass classification"""
    name = "Random Forest w/ imputation"
    model_type = "random_forest"
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, n_estimators, max_depth, impute_strategy,
                 percent_features, number_features, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=impute_strategy)
        enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)

        # estimator = RandomForestClassifier(random_state=random_state,
        #                                    n_estimators=n_estimators,
        #                                    max_depth=max_depth,
        #                                    n_jobs=n_jobs)
        estimator = RandomForestClassifier(random_state=random_state,
                                           n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           n_jobs=n_jobs)._component_obj
        feature_selection = SelectFromModel(
            estimator=estimator,
            max_features=max(1, int(percent_features * number_features)),
            threshold=-np.inf
        )

        self.pipeline = Pipeline(
            [("encoder", enc),
             ("imputer", imputer),
             ("feature_selection", feature_selection),
             ("estimator", estimator)]
        )

        super().__init__(objective=objective, random_state=random_state)

    @property
    def feature_importances(self):
        """Return feature importances. Feature dropped by feaure selection are excluded"""
        indices = self.pipeline["feature_selection"].get_support(indices=True)
        feature_names = list(map(lambda i: self.input_feature_names[i], indices))
        importances = list(zip(feature_names, self.pipeline["estimator"].feature_importances_))
        importances.sort(key=lambda x: -abs(x[1]))

        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df
