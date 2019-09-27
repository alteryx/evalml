import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt.space import Real

from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes
from evalml.models.model_types import ModelTypes


class LogisticRegressionPipeline(PipelineBase):
    """Logistic Regression Pipeline for both binary and multiclass classification"""
    name = "LogisticRegression w/ imputation + scaling"
    model_type = ModelTypes.LINEAR_MODEL
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }

    def __init__(self, objective, penalty, C, impute_strategy,
                 number_features, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=impute_strategy)
        enc = ce.OneHotEncoder(use_cat_names=True, return_df=True)

        estimator = LogisticRegression(random_state=random_state,
                                       penalty=penalty,
                                       C=C,
                                       multi_class='auto',
                                       solver="lbfgs",
                                       n_jobs=-1)

        self.pipeline = Pipeline(
            [("encoder", enc),
             ("imputer", imputer),
             ("scaler", StandardScaler()),
             ("estimator", estimator)]
        )

        super().__init__(objective=objective, random_state=random_state)

    @property
    def feature_importances(self):
        """Return feature importances. Feature dropped by feaure selection are excluded"""
        coef_ = self.pipeline["estimator"].coef_

        # binary classification case
        if len(coef_) <= 2:
            importances = list(zip(self.input_feature_names, coef_[0]))
            importances.sort(key=lambda x: -abs(x[1]))
        else:
            # mutliclass classification case
            importances = list(zip(self.input_feature_names, np.linalg.norm(coef_, axis=0, ord=2)))
            importances.sort(key=lambda x: -(x[1]))

        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df
