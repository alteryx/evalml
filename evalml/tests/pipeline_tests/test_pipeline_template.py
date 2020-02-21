import pytest

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineTemplate
from evalml.pipelines.components import LogisticRegressionClassifier
from evalml.problem_types import ProblemTypes


@pytest.fixture
def pipeline_template():
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Logistic Regression Classifier']
    supported_problem_types = ['binary', 'multiclass']
    return PipelineTemplate(component_graph=component_graph, supported_problem_types=supported_problem_types)


def test_init(pipeline_template):
    pipeline_template = pipeline_template

    assert pipeline_template.supported_problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
    assert pipeline_template.model_family == ModelTypes.LINEAR_MODEL
    assert pipeline_template.name == 'Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder'
    assert isinstance(pipeline_template.estimator, LogisticRegressionClassifier)


def test_problem_type_validation(pipeline_template):
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Logistic Regression Classifier']
    supported_problem_types = ['regression']
    with pytest.raises(ValueError):
        return PipelineTemplate(component_graph=component_graph, supported_problem_types=supported_problem_types)


def test_description(pipeline_template, capsys):
    pipeline_template.description
    description, _ = capsys.readouterr()

    assert "Supported Problem Types: Binary Classification, Multiclass Classification" in description
    assert "Model Family: Linear Model" in description
    assert "Simple Imputer" in description
    assert "One Hot Encoder" in description
    assert "Logistic Regression Classifier" in description
