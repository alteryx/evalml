from evalml.pipelines.classification_pipeline import ClassificationPipeline


class MulticlassClassificationPipeline(ClassificationPipeline):
    problem_types = ['multiclass']

