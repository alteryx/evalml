from evalml.pipelines import PipelineBase


class LogisticRegressionPipeline(PipelineBase):
    """Logistic Regression Pipeline for both binary and multiclass classification"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Logistic Regression Classifier']
    problem_types = ['binary', 'multiclass']
