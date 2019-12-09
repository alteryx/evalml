import numpy as np
import plotly.graph_objects as go
import sklearn.metrics
from scipy import interp

from evalml.problem_types import ProblemTypes


class PipelineSearchPlots:
    """Plots for the AutoClassifier/AutoRegressor class.
    """

    def __init__(self, data):
        """Make plots for the AutoClassifier/AutoRegressor class.

        Args:
            data (AutoClassifier or AutoRegressor): Automated pipeline search object
        """
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
            raise RuntimeError("You must first call fit() to generate ROC data.")

        if pipeline_id not in results:
            raise RuntimeError("Pipeline {} not found".format(pipeline_id))

        pipeline_results = results[pipeline_id]
        cv_data = pipeline_results["cv_data"]
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        roc_aucs = []
        fpr_tpr_data = []

        for fold in cv_data:
            fpr = fold["all_objective_scores"]["ROC"][0]
            tpr = fold["all_objective_scores"]["ROC"][1]
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

        layout = go.Layout(title={'text': 'Receiver Operating Characteristic of<br>{} w/ ID={}'.format(pipeline_name, pipeline_id)},
                           xaxis={'title': 'False Positive Rate', 'range': [-0.05, 1.05]},
                           yaxis={'title': 'True Positive Rate', 'range': [-0.05, 1.05]})
        data = []
        for fold_num, fold in enumerate(fpr_tpr_data):
            fpr = fold[0]
            tpr = fold[1]
            roc_auc = roc_aucs[fold_num]
            data.append(go.Scatter(x=fpr, y=tpr,
                                   name='ROC fold %d (AUC = %0.2f)' % (fold_num, roc_auc),
                                   mode='lines+markers'))

        data.append(go.Scatter(x=mean_fpr, y=mean_tpr,
                               name='Mean ROC (AUC = %0.2f &plusmn; %0.2f)' % (mean_auc, std_auc),
                               line=dict(width=3)))
        data.append(go.Scatter(x=[0, 1], y=[0, 1],
                               name='Chance',
                               line=dict(dash='dash')))

        figure = go.Figure(layout=layout, data=data)
        return figure

    def get_confusion_matrix_data(self, pipeline_id):
        """Gets data that can be used to create a confusion matrix plot.

        Returns:
            List containing information used to generate a confusion matrix plot. Each element in the list contains the confusion matrix data for that fold.
        """
        if self.data.problem_type not in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
            raise RuntimeError("Confusion matrix plots can only be generated for classification problems.")

        results = self.data.results['pipeline_results']
        if len(results) == 0:
            raise RuntimeError("You must first call fit() to generate confusion matrix data.")

        if pipeline_id not in results:
            raise RuntimeError("Pipeline {} not found".format(pipeline_id))

        pipeline_results = results[pipeline_id]
        cv_data = pipeline_results["cv_data"]

        confusion_matrix_data = []
        for fold in cv_data:
            confusion_matrix_data.append(fold["all_objective_scores"]["Confusion Matrix"])
        return confusion_matrix_data

    def generate_confusion_matrix(self, pipeline_id, fold_num=None):
        """Generate confusion matrix plot for a given pipeline using the data returned from get_confusion_matrix_data().

        Returns:
            plotly.Figure representing the confusion matrix plot generated

        """
        data = self.get_confusion_matrix_data(pipeline_id)
        results = self.data.results['pipeline_results']
        pipeline_name = results[pipeline_id]["pipeline_name"]
        # defaults to last fold if none specified. May need to think of better approach.
        if fold_num is None:
            fold_num = -1

        conf_mat = data[fold_num]
        labels = conf_mat.columns

        layout = go.Layout(title={'text': 'Confusion matrix of<br>{} w/ ID={}'.format(pipeline_name, pipeline_id)},
                           xaxis={'title': 'Predicted Label', 'tickvals': labels},
                           yaxis={'title': 'True Label', 'tickvals': labels})
        figure = go.Figure(data=go.Heatmap(x=labels, y=labels, z=conf_mat,
                                           hovertemplate='<b>True</b>: %{y}' +
                                                         '<br><b>Predicted</b>: %{x}' +
                                                         '<br><b>Number of times</b>: %{z}' +
                                                         '<extra></extra>'),  # necessary to remove unwanted trace info
                           layout=layout)
        return figure
