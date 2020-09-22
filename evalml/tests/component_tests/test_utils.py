import inspect

import pytest
from sklearn.utils.estimator_checks import check_estimator

from evalml.exceptions import MissingComponentError
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline, RegressionPipeline
from evalml.pipelines.components import ComponentBase, RandomForestClassifier
from evalml.pipelines.components.utils import (
    _all_estimators,
    all_components,
    handle_component_class,
    scikit_learn_wrapped_estimator
)
from evalml.pipelines.utils import make_pipeline_from_components
from evalml.problem_types import ProblemTypes


def test_all_components(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(all_components()) == 25
    else:
        assert len(all_components()) == 30


def test_handle_component_class_names():
    for cls in all_components():
        cls_ret = handle_component_class(cls)
        assert inspect.isclass(cls_ret)
        assert issubclass(cls_ret, ComponentBase)
        name_ret = handle_component_class(cls.name)
        assert inspect.isclass(name_ret)
        assert issubclass(name_ret, ComponentBase)

    invalid_name = 'This Component Does Not Exist'
    with pytest.raises(MissingComponentError, match='Component "This Component Does Not Exist" was not found'):
        handle_component_class(invalid_name)

    class NonComponent:
        pass
    with pytest.raises(ValueError):
        handle_component_class(NonComponent())


def test_scikit_learn_wrapper_invalid_problem_type():
    evalml_pipeline = make_pipeline_from_components([RandomForestClassifier()], ProblemTypes.MULTICLASS)
    evalml_pipeline.problem_type = None
    with pytest.raises(ValueError):
        scikit_learn_wrapped_estimator(evalml_pipeline)


def test_scikit_learn_wrapper(X_y_binary, X_y_multi, X_y_regression):
    for estimator in [estimator for estimator in _all_estimators() if estimator.model_family != ModelFamily.ENSEMBLE]:
        for problem_type in estimator.supported_problem_types:
            if problem_type == ProblemTypes.BINARY:
                X, y = X_y_binary
            elif problem_type == ProblemTypes.MULTICLASS:
                X, y = X_y_multi
            elif problem_type == ProblemTypes.REGRESSION:
                X, y = X_y_regression

            evalml_pipeline = make_pipeline_from_components([estimator()], problem_type)
            scikit_estimator = scikit_learn_wrapped_estimator(evalml_pipeline)
            # check_estimator(scikit_estimator)
            scikit_estimator.fit(X, y)
            print (scikit_estimator.predict(X))
            if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
                print (scikit_estimator.predict_proba(X))
