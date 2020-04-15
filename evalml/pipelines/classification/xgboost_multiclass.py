from evalml.pipelines import MulticlassClassificationPipeline


class XGBoostMulticlassPipeline(MulticlassClassificationPipeline):
    """XGBoost Pipeline for multiclass classification"""
    custom_name = "XGBoost Classifier w/ One Hot Encoder + Simple Imputer + RF Classifier Select From Model"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'XGBoost Classifier']

    def __init__(self, parameters, random_state=0):
        super().__init__(parameters=parameters,
                         random_state=random_state)
