from evalml.pipelines import RegressionPipeline


class XGBoostRegressionPipeline(RegressionPipeline):
    """XGBoost Pipeline for regression problems"""
    _name = "XGBoost Regression Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'XGBoost Regressor']
