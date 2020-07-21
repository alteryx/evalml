import pytest

from evalml.pipelines.explanations._user_interface import _make_single_prediction_table, _make_rows


@pytest.mark.parametrize("values, k, answer", [({"a": [0.2], "b": [0.1]}, 3, [["a", "++"], ["b", "+"]]),
                                               ({"a": [0.3], "b": [-0.9], "c": [0.5],
                                                 "d": [0.33], "e": [-0.67], "f": [-0.2],
                                                 "g": [0.71]}, 3,
                                                [["g", "++++"], ["c", "+++"], ["d", "++"],
                                                ["f", "--"], ["e", "----"], ["b", "-----"]]),
                                               ({"a": [1.0], "f": [-1.0], "e": [0.0]}, 5,
                                                [["a", "+++++"], ["e", "+"], ["f", "-----"]])])
def test_make_rows(values, k, answer):
    assert _make_rows(values, k) == answer
    # Subtracting two because a header and a line under the header are included in the table.
    assert len(_make_single_prediction_table(values, k).splitlines()) - 2 == len(answer)


