from evalml.pipelines import PipelineBase


class RegressionPipeline(PipelineBase):
    supported_problem_types = ['regression']
