from evalml.pipelines import BinaryClassificationPipeline


class RFBinaryClassificationPipeline(BinaryClassificationPipeline):
    """Random Forest Pipeline for binary classification."""
    custom_name = "Random Forest Binary Classification Pipeline"
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Random Forest Classifier']
