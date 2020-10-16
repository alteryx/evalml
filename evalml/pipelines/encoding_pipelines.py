from itertools import product

from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline
from evalml.pipelines.multiclass_classification_pipeline import MulticlassClassificationPipeline
from evalml.problem_types import ProblemTypes, handle_problem_types


def _make_encoding_pipeline(pipeline_class, encoder, estimator):

    pipeline_name = f"{estimator} with {encoder}"

    c_graph = [encoder, estimator] if encoder else [estimator]

    class EncodingPipeline(pipeline_class):
        custom_name = pipeline_name
        component_graph = c_graph

    return EncodingPipeline


def get_encoding_pipelines(problem_type):
    problem_type = handle_problem_types(problem_type)
    pipeline_class = MulticlassClassificationPipeline

    if problem_type == ProblemTypes.BINARY:
        pipeline_class = BinaryClassificationPipeline

    pipelines = []
    encoders = ["One Hot Encoder", "Ordinal Encoder", "Binary Encoder", "Sum Encoder"]
    estimators = ["Random Forest Classifier", "XGBoost Classifier", "Logistic Regression Classifier"]
    for encoder, estimator in product(encoders, estimators):
        pipelines.append(_make_encoding_pipeline(pipeline_class, encoder, estimator))

    return pipelines + [_make_encoding_pipeline(pipeline_class, "One Hot Encoder", "LightGBM Classifier"),
                        _make_encoding_pipeline(pipeline_class, None, "LightGBM Classifier")]
