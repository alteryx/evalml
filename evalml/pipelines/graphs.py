import os.path

import plotly.graph_objects as go

from evalml.utils.gen_utils import import_or_raise


def make_pipeline_graph(component_list, graph_name, filepath=None):
    """Create a graph of the pipeline, in a format similar to a UML diagram.

    Arguments:
        pipelne (PipelineBase) : The pipeline to make a graph of.
        filepath (str, optional) : Path to where the graph should be saved. If set to None (as by default), the graph will not be saved.

    Returns:
        graphviz.Digraph : Graph object that can directly be displayed in Jupyter notebooks.
    """
    graphviz = import_or_raise('graphviz', error_msg='Please install graphviz to visualize pipelines.')

    # Try rendering a dummy graph to see if a working backend is installed
    try:
        graphviz.Digraph().pipe()
    except graphviz.backend.ExecutableNotFound:
        raise RuntimeError(
            "To graph entity sets, a graphviz backend is required.\n" +
            "Install the backend using one of the following commands:\n" +
            "  Mac OS: brew install graphviz\n" +
            "  Linux (Ubuntu): sudo apt-get install graphviz\n" +
            "  Windows: conda install python-graphviz\n"
        )

    graph_format = None
    if filepath:
        # Explicitly cast to str in case a Path object was passed in
        filepath = str(filepath)
        path_and_name, graph_format = os.path.splitext(filepath)
        graph_format = graph_format[1:].lower()  # ignore the dot
        supported_filetypes = graphviz.backend.FORMATS
        if graph_format not in supported_filetypes:
            raise ValueError(("Unknown format '{}'. Make sure your format is one of the " +
                              "following: {}").format(graph_format, supported_filetypes))

    # Initialize a new directed graph
    graph = graphviz.Digraph(name=graph_name, format=graph_format,
                             graph_attr={'splines': 'ortho'})
    graph.attr(rankdir='LR')

    # Draw components
    for component in component_list:
        label = '%s\l' % (component.name)  # noqa: W605
        if len(component.parameters) > 0:
            parameters = '\l'.join([key + ' : ' + "{:0.2f}".format(val) if (isinstance(val, float))
                                    else key + ' : ' + str(val)
                                    for key, val in component.parameters.items()])  # noqa: W605
            label = '%s |%s\l' % (component.name, parameters)  # noqa: W605
        graph.node(component.name, shape='record', label=label)

    # Draw edges
    for i in range(len(component_list[:-1])):
        graph.edge(component_list[i].name, component_list[i + 1].name)

    if filepath:
        graph.render(filepath, cleanup=True)

    return graph


def make_feature_importance_graph(feature_importances, show_all_features=False):
    """Create and return a bar graph of the pipeline's feature importances

    Arguments:
        feature_importances (pd.DataFrame) : The pipeline with which to compute feature importances.
        show_all_features (bool, optional) : If true, graph features with an importance value of zero. Defaults to false.

    Returns:
        plotly.Figure, a bar graph showing features and their importances
    """
    feat_imp = feature_importances
    feat_imp['importance'] = abs(feat_imp['importance'])

    if not show_all_features:
        # Remove features with zero importance
        feat_imp = feat_imp[feat_imp['importance'] != 0]

    # List is reversed to go from ascending order to descending order
    feat_imp = feat_imp.iloc[::-1]

    title = 'Feature Importances'
    subtitle = 'May display fewer features due to feature selection'
    data = [go.Bar(
        x=feat_imp['importance'],
        y=feat_imp['feature'],
        orientation='h'
    )]

    layout = {
        'title': '{0}<br><sub>{1}</sub>'.format(title, subtitle),
        'height': 800,
        'xaxis_title': 'Feature Importance',
        'yaxis_title': 'Feature',
        'yaxis': {
            'type': 'category'
        }
    }

    fig = go.Figure(data=data, layout=layout)
    return fig
