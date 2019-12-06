

class PipelinePlots:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, to_file=None):
        """
        Create a UML diagram-ish graph of our pipeline.


        Args:
            to_file (str, optional) : Path to where the plot should be saved. If set to None (as by default), the plot will not be saved.

        Returns:
            graphviz.Digraph : Graph object that can directly be displayed in Jupyter notebooks.
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError('Please install graphviz to visualize pipelines.')

        # Try rendering a dummy graph to see if a working backend is installed
        try:
            graphviz.Digraph().pipe()
        except graphviz.backend.ExecutableNotFound:
            raise RuntimeError(
                "To plot entity sets, a graphviz backend is required.\n" +
                "Install the backend using one of the following commands:\n" +
                "  Mac OS: brew install graphviz\n" +
                "  Linux (Ubuntu): sudo apt-get install graphviz\n" +
                "  Windows: conda install python-graphviz\n"
            )

        if to_file:
            # Explicitly cast to str in case a Path object was passed in
            to_file = str(to_file)
            split_path = to_file.split('.')
            if len(split_path) < 2:
                raise ValueError("Please use a file extension like '.pdf'" +
                                 " so that the format can be inferred")

            format = split_path[-1]
            valid_formats = graphviz.backend.FORMATS
            if format not in valid_formats:
                raise ValueError("Unknown format. Make sure your format is" +
                                 " amongst the following: %s" % valid_formats)
        else:
            format = None

        # Initialize a new directed graph
        graph = graphviz.Digraph(name=self.pipeline.name, format=format,
                                 graph_attr={'splines': 'ortho'})
        graph.attr(rankdir='LR')

        # Draw components
        for component in self.pipeline.component_list:
            label = '%s\l' % (component.name)  # noqa: W605
            if len(component.parameters) > 0:
                parameters = '\l'.join([key + ' : ' + "{:0.2f}".format(val) if (isinstance(val, float))
                                        else key + ' : ' + str(val)
                                        for key, val in component.parameters.items()])  # noqa: W605
                label = '%s |%s\l' % (component.name, parameters)  # noqa: W605
            graph.node(component.name, shape='record', label=label)

        # Draw edges
        for i in range(len(self.pipeline.component_list[:-1])):
            graph.edge(self.pipeline.component_list[i].name, self.pipeline.component_list[i + 1].name)

        if to_file:
            # Graphviz always appends the format to the file name, so we need to
            # remove it manually to avoid file names like 'file_name.pdf.pdf'
            offset = len(format) + 1  # Add 1 for the dot
            output_path = to_file[:-offset]
            graph.render(output_path, cleanup=True)

        return graph

    def feature_importances(self):
        import plotly.graph_objects as go

        feat_imp = self.pipeline.feature_importances
        feat_imp['importance'] = abs(feat_imp['importance'])
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
