from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    LogisticRegressionClassifier,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler
)


class LogisticRegressionPipeline(PipelineBase):
    """Logistic Regression Pipeline for both binary and multiclass classification"""

    def __init__(self, objective, penalty, C, impute_strategy,
                 number_features, n_jobs=-1, random_state=0):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        scaler = StandardScaler()
        estimator = LogisticRegressionClassifier(random_state=random_state,
                                                 penalty=penalty,
                                                 C=C,
                                                 n_jobs=-1)

        super().__init__(objective=objective,
                         component_list=[enc, imputer, scaler, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)
