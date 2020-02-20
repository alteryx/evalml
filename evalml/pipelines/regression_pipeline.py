
from .pipeline_plots import PipelinePlots

from evalml.pipelines import PipelineBase


class RegressionPipeline(PipelineBase):

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelinePlots
