from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes


class RegressionPipeline(PipelineBase):
    """Pipeline subclass for all regression pipelines."""
    problem_type = ProblemTypes.REGRESSION
