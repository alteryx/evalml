from evalml.pipelines import BinaryClassificationPipeline


class ENBinaryPipeline(BinaryClassificationPipeline):
    """Elastic Net Pipeline for binary classification problems."""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Elastic Net Classifier']
