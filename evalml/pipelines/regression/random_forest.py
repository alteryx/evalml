from skopt.space import Integer, Real

from evalml.pipelines import RegressionPipeline


class RFRegressionPipeline(RegressionPipeline):
    """Random Forest Pipeline for regression problems"""
    _name = "Random Forest Regression Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Regressor Select From Model', 'Random Forest Regressor']
    problem_types = ['regression']

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }
