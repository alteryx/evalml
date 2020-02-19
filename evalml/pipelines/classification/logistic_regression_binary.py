from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import BinaryClassificationPipeline
from evalml.pipelines.components import (
    LogisticRegressionClassifier,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler
)
from evalml.problem_types import ProblemTypes


class LogisticRegressionBinaryPipeline(BinaryClassificationPipeline):
    """Logistic Regression Pipeline for binary classification"""
    name = "Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    problem_types = [ProblemTypes.BINARY]

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }

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
