from evalml.pipelines.components import components_dict

def test_print():
    print(sorted(components_dict().keys()))
    assert len(components_dict()) == 8