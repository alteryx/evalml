
from evalml.pipelines import RegressionPipeline


class LinearRegressionPipeline(RegressionPipeline):
    """Linear Regression Pipeline for regression problems"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Linear Regressor']
    supported_problem_types = ['regression']
