from evalml.pipelines import BinaryClassificationPipeline


class BaselineBinaryPipeline(BinaryClassificationPipeline):
    """Baseline Pipeline for binary classification."""
    custom_name = "Baseline Classification Pipeline"
    component_graph = ["Baseline Classifier"]


class ModeBaselineBinaryPipeline(BinaryClassificationPipeline):
    """Mode Baseline Pipeline for binary classification."""
    custom_name = "Mode Baseline Binary Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = {"strategy": ["mode"]}
