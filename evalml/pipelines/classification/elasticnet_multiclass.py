from evalml.pipelines import MulticlassClassificationPipeline


class ENMulticlassPipeline(MulticlassClassificationPipeline):
    """Elastic Net Pipeline for multiclass classification problems"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Elastic Net Classifier']
