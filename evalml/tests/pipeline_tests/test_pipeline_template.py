import pytest

from evalml.pipelines import PipelineTemplate

@pytest.fixture
def pipeline_template():
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Logistic Regression Classifier']
    supported_problem_types = ['binary', 'multiclass']
    return PipelineTemplate(component_graph=component_graph, supported_problem_types=supported_problem_types)

def test_init(pipeline_template):
    pipeline_template = pipeline_template