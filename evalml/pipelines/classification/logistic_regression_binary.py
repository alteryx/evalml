from evalml.pipelines import BinaryClassificationPipeline


class LogisticRegressionBinaryPipeline(BinaryClassificationPipeline):
    """Logistic Regression Pipeline for binary classification."""
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']
