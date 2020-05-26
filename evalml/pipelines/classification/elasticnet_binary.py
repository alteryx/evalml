from evalml.pipelines import BinaryClassificationPipeline


class ENRegressionPipeline(BinaryClassificationPipeline):
    """Elastic Net Pipeline for binary classification problems"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Elastic Net Classifier']
