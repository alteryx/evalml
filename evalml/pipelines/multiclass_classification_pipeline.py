from evalml.pipelines.classification_pipeline import ClassificationPipeline


class MulticlassClassificationPipeline(ClassificationPipeline):

    threshold_selection_split = False  # primary difference between binary and multiclass
