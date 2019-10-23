from evalml.pipelines.components import LinearRegressor, components_dict, handle_component

def test_components_dict():
    assert len(components_dict()) == 8


def test_handle_component():
    component_strs = ['Linear Regressor']
    components = [LinearRegressor]
    
    for c in zip(component_strs, components):
        assert isinstance(handle_component(c[0]), c[1])
