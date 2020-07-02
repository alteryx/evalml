import inspect

import pytest

from evalml.exceptions import MissingComponentError
from evalml.pipelines.components import ComponentBase
from evalml.pipelines.components.utils import (
    all_components,
    handle_component_class
)


def test_all_components(has_minimal_dependencies):
    if has_minimal_dependencies:
        assert len(all_components) == 20
    else:
        assert len(all_components) == 24


def test_handle_component_class_names():
    for cls in all_components:
        cls_ret = handle_component_class(cls)
        assert inspect.isclass(cls_ret)
        assert issubclass(cls_ret, ComponentBase)
        name_ret = handle_component_class(cls.name)
        assert inspect.isclass(name_ret)
        assert issubclass(name_ret, ComponentBase)

    for cls in all_components:
        obj = cls()
        with pytest.raises(ValueError, match='component_graph may only contain str or ComponentBase subclasses, not'):
            handle_component_class(obj)

    invalid_name = 'This Component Does Not Exist'
    with pytest.raises(MissingComponentError, match='Component "This Component Does Not Exist" was not found'):
        handle_component_class(invalid_name)

    class NonComponent:
        pass
    with pytest.raises(ValueError):
        handle_component_class(NonComponent())
