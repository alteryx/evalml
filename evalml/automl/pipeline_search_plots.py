import numpy as np
import sklearn.metrics
from scipy import interp

from evalml.problem_types import ProblemTypes
from evalml.utils import import_or_raise, normalize_confusion_matrix


class SearchIterationPlot():
    def __init__(self, data, show_plot=True):
        self._go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
        self.data = data
        self.best_score_by_iter_fig = None
        self.curr_iteration_scores = list()
        self.best_iteration_scores = list()

        title = 'Pipeline Search: Iteration vs. {}<br><sub>Gray marker indicates the score at current iteration</sub>'.format(self.data.objective.name)
        data = [
            self._go.Scatter(x=[], y=[], mode='lines+markers', name='Best Score'),
            self._go.Scatter(x=[], y=[], mode='markers', name='Iter score', marker={'color': 'gray'})
        ]
        layout = {
            'title': title,
            'xaxis': {
                'title': 'Iteration',
                'rangemode': 'tozero'
            },
            'yaxis': {
                'title': 'Score'
            }
        }
        self.best_score_by_iter_fig = self._go.FigureWidget(data, layout)
        self.best_score_by_iter_fig.update_layout(showlegend=False)
        self.update()

    def update(self):
        if len(self.data.results['search_order']) > 0 and len(self.data.results['pipeline_results']) > 0:
            iter_idx = self.data.results['search_order']
            pipeline_res = self.data.results['pipeline_results']
            iter_scores = [pipeline_res[i]['score'] for i in iter_idx]

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
                    if self.data.objective.greater_is_better and score > curr_best \
                            or not self.data.objective.greater_is_better and score < curr_best:
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
    """Plots for the AutoClassificationSearch/AutoRegressionSearch class.
    """

    def __init__(self, data):
        """Make plots for the AutoClassificationSearch/AutoRegressionSearch class.

        Args:
            data (AutoClassificationSearch or AutoRegressionSearch): Automated pipeline search object
        """
        self._go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
        self.data = data

    def get_roc_data(self, pipeline_id):
        """Gets data that can be used to create a ROC plot.

        Returns:
            Dictionary containing metrics used to generate an ROC plot.
        """
        if self.data.problem_type != ProblemTypes.BINARY:
            raise RuntimeError("ROC plots can only be generated for binary classification problems.")

        results = self.data.results['pipeline_results']
        if len(results) == 0:
            raise RuntimeError("You must first call search() to generate ROC data.")

        if pipeline_id not in results:
            raise RuntimeError("Pipeline {} not found".format(pipeline_id))

        pipeline_results = results[pipeline_id]
        plot_data = pipeline_results["plot_data"]
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        roc_aucs = []
        fpr_tpr_data = []

        for fold in plot_data:
            fpr = fold["ROC"][0]
            tpr = fold["ROC"][1]
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            roc_aucs.append(roc_auc)
            fpr_tpr_data.append((fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(roc_aucs)

        roc_data = {"fpr_tpr_data": fpr_tpr_data,
                    "mean_fpr": mean_fpr,
                    "mean_tpr": mean_tpr,
                    "roc_aucs": roc_aucs,
                    "mean_auc": mean_auc,
                    "std_auc": std_auc}
        return roc_data

    def generate_roc_plot(self, pipeline_id):
        """Generate Receiver Operating Characteristic (ROC) plot for a given pipeline using cross-validation
        using the data returned from get_roc_data().

        Returns:
            plotly.Figure representing the ROC plot generated

        """
        roc_data = self.get_roc_data(pipeline_id)
        fpr_tpr_data = roc_data["fpr_tpr_data"]
        roc_aucs = roc_data["roc_aucs"]
        mean_fpr = roc_data["mean_fpr"]
        mean_tpr = roc_data["mean_tpr"]
        mean_auc = roc_data["mean_auc"]
        std_auc = roc_data["std_auc"]

        results = self.data.results['pipeline_results']
        pipeline_name = results[pipeline_id]["pipeline_name"]

        layout = self._go.Layout(title={'text': 'Receiver Operating Characteristic of<br>{} w/ ID={}'.format(pipeline_name, pipeline_id)},
                                 xaxis={'title': 'False Positive Rate', 'range': [-0.05, 1.05]},
                                 yaxis={'title': 'True Positive Rate', 'range': [-0.05, 1.05]})
        data = []
        for fold_num, fold in enumerate(fpr_tpr_data):
            fpr = fold[0]
            tpr = fold[1]
            roc_auc = roc_aucs[fold_num]
            data.append(self._go.Scatter(x=fpr, y=tpr,
                                         name='ROC fold %d (AUC = %0.2f)' % (fold_num, roc_auc),
                                         mode='lines+markers'))

        data.append(self._go.Scatter(x=mean_fpr, y=mean_tpr,
                                     name='Mean ROC (AUC = %0.2f &plusmn; %0.2f)' % (mean_auc, std_auc),
                                     line=dict(width=3)))
        data.append(self._go.Scatter(x=[0, 1], y=[0, 1],
                                     name='Chance',
                                     line=dict(dash='dash')))

        figure = self._go.Figure(layout=layout, data=data)
        return figure

    def get_confusion_matrix_data(self, pipeline_id, normalize=None):
        """Gets data that can be used to create a confusion matrix plot.

        Arguments:
            pipeline_id (int): ID of pipeline to get confusion matrix data for
            normalize ({'true', 'pred', 'all', None}): Option to normalize over the rows ('true'), columns ('pred') or all ('all') values. If option is None, returns original confusion matrix. Defaults to 'true'.

        Returns:
            List containing information used to generate a confusion matrix plot. Each element in the list contains the confusion matrix data for that fold.
        """
        if self.data.problem_type not in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
            raise RuntimeError("Confusion matrix plots can only be generated for classification problems.")

        results = self.data.results['pipeline_results']
        if len(results) == 0:
            raise RuntimeError("You must first call search() to generate confusion matrix data.")

        if pipeline_id not in results:
            raise RuntimeError("Pipeline {} not found".format(pipeline_id))

        pipeline_results = results[pipeline_id]
        plot_data = pipeline_results["plot_data"]

        confusion_matrix_data = []
        for fold in plot_data:
            conf_mat = fold["Confusion Matrix"]
            # reverse columns in confusion matrix to change axis order to match sklearn's
            conf_mat = conf_mat.iloc[:, ::-1]
            if normalize is not None:
                conf_mat = normalize_confusion_matrix(conf_mat, option=normalize)
            confusion_matrix_data.append(conf_mat)
        return confusion_matrix_data

    def generate_confusion_matrix(self, pipeline_id, fold_num=None, normalize=None):
        """Generate confusion matrix plot for a given pipeline using the data returned from get_confusion_matrix_data().

        Arguments:
            pipeline_id (int): ID of pipeline to get confusion matrix data for
            fold_num (int): Fold number of pipeline to get confusion matrix data for
            option ({'true', 'pred', 'all', None}): Option to normalize over the rows ('true'), columns ('pred') or all ('all') values. If option is None, returns original confusion matrix. Defaults to 'true'.

        Returns:
            plotly.Figure representing the confusion matrix plot generated

        """
        data = self.get_confusion_matrix_data(pipeline_id, normalize=None)
        if normalize is not None:
            data_normalized = self.get_confusion_matrix_data(pipeline_id, normalize=normalize)
        else:
            data_normalized = self.get_confusion_matrix_data(pipeline_id, normalize='true')

        results = self.data.results['pipeline_results']
        pipeline_name = results[pipeline_id]["pipeline_name"]
        # defaults to last fold if none specified. May need to think of better approach.
        if fold_num is None:
            fold_num = -1

        conf_mat = data[fold_num]
        conf_mat_normalized = data_normalized[fold_num]

        labels = conf_mat.columns
        reversed_labels = labels[::-1]

        title_text = 'Confusion matrix of<br>{} w/ ID={}'.format(pipeline_name, pipeline_id)
        z_data = conf_mat
        custom_data = conf_mat_normalized
        hover_text = '<br><b>Number of times</b>: %{z}' + '<br><b>Normalized</b>: %{customdata:.3f} <br>'

        if normalize is not None:
            title_text = 'Normalized confusion matrix of<br>{} w/ ID={}'.format(pipeline_name, pipeline_id)
            z_data = conf_mat_normalized
            custom_data = conf_mat
            hover_text = '<br><b>Number of times</b>: %{customdata}' + '<br><b>Normalized</b>: %{z:.3f} <br>'

        layout = self._go.Layout(title={'text': title_text},
                                 xaxis={'title': 'Predicted Label', 'type': 'category', 'tickvals': labels},
                                 yaxis={'title': 'True Label', 'type': 'category', 'tickvals': reversed_labels})
        figure = self._go.Figure(data=self._go.Heatmap(x=labels, y=reversed_labels, z=z_data,
                                                       customdata=custom_data,
                                                       hovertemplate='<b>True</b>: %{y}' +
                                                       '<br><b>Predicted</b>: %{x}' +
                                                       hover_text +
                                                       '<extra></extra>',  # necessary to remove unwanted trace info
                                                       colorscale='Blues'),
                                 layout=layout)
        return figure

    def search_iteration_plot(self, interactive_plot=False):
        """Shows a plot of the best score at each iteration using data gathered during training.

        Returns:
            plot
        """
        if not interactive_plot:
            plot_obj = SearchIterationPlot(self.data)
            return self._go.Figure(plot_obj.best_score_by_iter_fig)
        try:
            ipython_display = import_or_raise("IPython.display", error_msg="Cannot find dependency IPython.display")
            plot_obj = SearchIterationPlot(self.data)
            ipython_display.display(plot_obj.best_score_by_iter_fig)
            return plot_obj
        except ImportError:
            return self.search_iteration_plot(interactive_plot=False)
