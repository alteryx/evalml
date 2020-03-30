from evalml.pipelines import BinaryClassificationPipeline


class LogisticRegressionBinaryPipeline(BinaryClassificationPipeline):
    """Logistic Regression Pipeline for binary classification"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Logistic Regression Classifier']
    supported_problem_types = ['binary']
