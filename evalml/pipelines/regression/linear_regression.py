from evalml.pipelines import PipelineBase


class LinearRegressionPipeline(PipelineBase):
    """Linear Regression Pipeline for regression problems"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Linear Regressor']
    problem_types = ['regression']
