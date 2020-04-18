from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes


class RegressionPipeline(PipelineBase):
    problem_type = ProblemTypes.REGRESSION
