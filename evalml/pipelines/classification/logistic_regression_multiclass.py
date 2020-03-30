from evalml.pipelines import MulticlassClassificationPipeline


class LogisticRegressionMulticlassPipeline(MulticlassClassificationPipeline):
    """Logistic Regression Pipeline for multiclass classification"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Logistic Regression Classifier']
