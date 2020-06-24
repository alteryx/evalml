import inspect
import sys

import pytest

from evalml.pipelines.classification import *  # noqa: F401,F403
from evalml.pipelines.components import *  # noqa: F401,F403
from evalml.pipelines.regression import *  # noqa: F401,F403

all_components = inspect.getmembers(sys.modules['evalml.pipelines.components'], inspect.isclass)

regression_pipelines = inspect.getmembers(sys.modules['evalml.pipelines.regression'], inspect.isclass)
classification_pipelines = inspect.getmembers(sys.modules['evalml.pipelines.classification'], inspect.isclass)
all_pipelines = regression_pipelines + classification_pipelines


@pytest.mark.parametrize("class_name,cls", all_components)
def test_default_parameters(class_name, cls):

    # Can't instantiate these base classes to check the defaults.
    if class_name in {"Transformer", "FeatureSelector", "Estimator", "ComponentBase",
                      "CategoricalEncoder"}:
        assert True

    else:
        assert cls.default_parameters == cls().parameters, f"{class_name}'s default parameters don't match __init__."


@pytest.mark.parametrize("class_name,cls", all_pipelines)
def test_pipeline_default_parameters(class_name, cls):

    # Can't instantiate the base class to check the defaults.
    if class_name in {"PipelineBase"}:
        assert True

    else:
        assert cls.default_parameters == cls({}).parameters, f"{class_name}'s default parameters don't match __init__."
