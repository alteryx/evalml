from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from .components import Estimator, handle_component
from .pipeline_plots import PipelinePlots

from evalml.objectives import get_objective
from evalml.pipelines import PipelineBase
from evalml.utils import Logger


class RegressionPipeline(PipelineBase):

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelinePlots
