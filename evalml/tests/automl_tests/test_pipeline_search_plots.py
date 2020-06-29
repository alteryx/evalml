import pandas as pd
import pytest

from evalml.automl.pipeline_search_plots import SearchIterationPlot


def test_search_iteration_plot_class(X_y_binary):
    pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')

    class MockObjective:
        def __init__(self):
            self.name = 'Test Objective'
            self.greater_is_better = True

    class MockResults:
        def __init__(self):
            self.objective = MockObjective()
            self.results = {
                'pipeline_results': {
                    2: {
                        'score': 0.50
                    },
                    0: {
                        'score': 0.60
                    },
                    1: {
                        'score': 0.75
                    },
                },
                'search_order': [1, 2, 0]
            }
            self.rankings = pd.DataFrame({
                'score': [0.75, 0.60, 0.50]
            })

    mock_data = MockResults()
    plot = SearchIterationPlot(mock_data)

    # Check best score trace
    plot_data = plot.best_score_by_iter_fig.data[0]
    x = list(plot_data['x'])
    y = list(plot_data['y'])

    assert isinstance(plot, SearchIterationPlot)
    assert x == [0, 1, 2]
    assert y == [0.60, 0.75, 0.75]

    # Check current score trace
    plot_data = plot.best_score_by_iter_fig.data[1]
    x = list(plot_data['x'])
    y = list(plot_data['y'])

    assert isinstance(plot, SearchIterationPlot)
    assert x == [0, 1, 2]
    assert y == [0.60, 0.75, 0.50]
