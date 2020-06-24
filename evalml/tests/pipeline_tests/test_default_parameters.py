import inspect
import sys

import pytest

from evalml.exceptions import MissingComponentError
from evalml.pipelines import PipelineBase
from evalml.pipelines.classification import *  # noqa: F401,F403
from evalml.pipelines.components import *  # noqa: F401,F403
from evalml.pipelines.regression import *  # noqa: F401,F403

all_components = inspect.getmembers(sys.modules['evalml.pipelines.components'], inspect.isclass)

regression_pipelines = inspect.getmembers(sys.modules['evalml.pipelines.regression'], inspect.isclass)
classification_pipelines = inspect.getmembers(sys.modules['evalml.pipelines.classification'], inspect.isclass)
all_pipelines = regression_pipelines + classification_pipelines + [("PipelineBase", PipelineBase)]


def cannot_check_because_base_or_not_installed(cls):

    if issubclass(cls, ComponentBase):  # noqa: F405
        def function(cls):
            cls().parameters
    else:
        def function(cls):
            cls({}).parameters
    try:
        function(cls)
    except (ImportError, TypeError, MissingComponentError):
        return True
    else:
        return False


@pytest.mark.parametrize("class_name,cls", all_components)
def test_default_parameters(class_name, cls):

    if cannot_check_because_base_or_not_installed(cls):
        pytest.skip(f"Skipping {class_name} because it is not installed or it is a base class.")

    assert cls.default_parameters == cls().parameters, f"{class_name}'s default parameters don't match __init__."


@pytest.mark.parametrize("class_name,cls", all_pipelines)
def test_pipeline_default_parameters(class_name, cls):

    if cannot_check_because_base_or_not_installed(cls):
        pytest.skip(f"Skipping {class_name} because it is not installed or it is a base class.")

    assert cls.default_parameters == cls({}).parameters, f"{class_name}'s default parameters don't match __init__."
