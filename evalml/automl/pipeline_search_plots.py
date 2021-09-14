"""Plots displayed during pipeline search."""
from evalml.utils import import_or_raise, jupyter_check


class SearchIterationPlot:
    """Search iteration plot.

    Args:
        results (dict): Dictionary of current results.
        objective (ObjectiveBase): Objective that AutoML is optimizing for.
    """

    def __init__(self, results, objective):
        self._go = import_or_raise(
            "plotly.graph_objects",
            error_msg="Cannot find dependency plotly.graph_objects",
        )

        if jupyter_check():
            import_or_raise("ipywidgets", warning=True)

        self.best_score_by_iter_fig = None
        self.curr_iteration_scores = list()
        self.best_iteration_scores = list()

        title = "Pipeline Search: Iteration vs. {}<br><sub>Gray marker indicates the score at current iteration</sub>".format(
            objective.name
        )
        data = [
            self._go.Scatter(x=[], y=[], mode="lines+markers", name="Best Score"),
            self._go.Scatter(
                x=[], y=[], mode="markers", name="Iter score", marker={"color": "gray"}
            ),
        ]
        layout = {
            "title": title,
            "xaxis": {"title": "Iteration", "rangemode": "tozero"},
            "yaxis": {"title": "Score"},
        }
        self.best_score_by_iter_fig = self._go.FigureWidget(data, layout)
        self.best_score_by_iter_fig.update_layout(showlegend=False)
        self.update(results, objective)
        self._go = None

    def update(self, results, objective):
        """Update the search plot."""
        if len(results["search_order"]) > 0 and len(results["pipeline_results"]) > 0:
            iter_idx = results["search_order"]
            pipeline_res = results["pipeline_results"]
            iter_scores = [pipeline_res[i]["mean_cv_score"] for i in iter_idx]

            iter_score_pairs = zip(iter_idx, iter_scores)
            iter_score_pairs = sorted(iter_score_pairs, key=lambda value: value[0])
            sorted_iter_idx, sorted_iter_scores = zip(*iter_score_pairs)

            # Create best score data
            best_iteration_scores = list()
            curr_best = None
            for score in sorted_iter_scores:
                if curr_best is None:
                    best_iteration_scores.append(score)
                    curr_best = score
                else:
                    if (
                        objective.greater_is_better
                        and score > curr_best
                        or not objective.greater_is_better
                        and score < curr_best
                    ):
                        best_iteration_scores.append(score)
                        curr_best = score
                    else:
                        best_iteration_scores.append(curr_best)

            # Update entire line plot
            best_score_trace = self.best_score_by_iter_fig.data[0]
            best_score_trace.x = sorted_iter_idx
            best_score_trace.y = best_iteration_scores

            curr_score_trace = self.best_score_by_iter_fig.data[1]
            curr_score_trace.x = sorted_iter_idx
            curr_score_trace.y = sorted_iter_scores


class PipelineSearchPlots:
    """Plots for the AutoMLSearch class during search.

    Args:
        results (dict): Dictionary of current results.
        objective (ObjectiveBase): Objective that AutoML is optimizing for.
    """

    def __init__(self, results, objective):
        self._go = import_or_raise(
            "plotly.graph_objects",
            error_msg="Cannot find dependency plotly.graph_objects",
        )
        self.results = results
        self.objective = objective

    def search_iteration_plot(self, interactive_plot=False):
        """Shows a plot of the best score at each iteration using data gathered during training.

        Args:
            interactive_plot (bool): Whether or not to show an interactive plot. Defaults to False.

        Returns:
            plot

        Raises:
            ValueError: If engine_str is not a valid engine.
        """
        if not interactive_plot:
            plot_obj = SearchIterationPlot(self.results, self.objective)
            return self._go.Figure(plot_obj.best_score_by_iter_fig)
        try:
            ipython_display = import_or_raise(
                "IPython.display", error_msg="Cannot find dependency IPython.display"
            )
            plot_obj = SearchIterationPlot(self.results, self.objective)
            ipython_display.display(plot_obj.best_score_by_iter_fig)
            return plot_obj
        except ImportError:
            return self.search_iteration_plot(interactive_plot=False)
