from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.problem_types import ProblemTypes


class MulticlassClassificationPipeline(ClassificationPipeline):
    """Pipeline subclass for all multiclass classification pipelines."""
    problem_type = ProblemTypes.MULTICLASS
