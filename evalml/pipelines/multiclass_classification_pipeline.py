"""Pipeline subclass for all multiclass classification pipelines."""
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.problem_types import ProblemTypes


class MulticlassClassificationPipeline(ClassificationPipeline):
    """Pipeline subclass for all multiclass classification pipelines.

    Args:
        component_graph (ComponentGraph, list, dict): ComponentGraph instance, list of components in order, or dictionary of components.
            Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters. Defaults to None.
        custom_name (str): Custom name for the pipeline. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> pipeline = MulticlassClassificationPipeline(component_graph=["Simple Imputer", "Logistic Regression Classifier"],
        ...                                             parameters={"Logistic Regression Classifier": {"penalty": "elasticnet",
        ...                                                                                            "solver": "liblinear"}},
        ...                                             custom_name="My Multiclass Pipeline")
        ...
        >>> assert pipeline.custom_name == "My Multiclass Pipeline"
        >>> assert pipeline.component_graph.component_dict.keys() == {'Simple Imputer', 'Logistic Regression Classifier'}

        The pipeline parameters will be chosen from the default parameters for every component, unless specific parameters
        were passed in as they were above.

        >>> assert pipeline.parameters == {
        ...     'Simple Imputer': {'impute_strategy': 'most_frequent', 'fill_value': None},
        ...     'Logistic Regression Classifier': {'penalty': 'elasticnet',
        ...                                        'C': 1.0,
        ...                                        'n_jobs': -1,
        ...                                        'multi_class': 'auto',
        ...                                        'solver': 'liblinear'}}
    """

    problem_type = ProblemTypes.MULTICLASS
    """ProblemTypes.MULTICLASS"""
