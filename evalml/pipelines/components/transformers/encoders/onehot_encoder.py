import category_encoders as ce

from .encoder import CategoricalEncoder

from evalml.pipelines.components import ComponentTypes


class OneHotEncoder(CategoricalEncoder):

    """Creates one-hot encoding for non-numeric data"""
    name = 'One Hot Encoder'
    component_type = ComponentTypes.CATEGORICAL_ENCODER
    _needs_fitting = True
    hyperparameter_ranges = {}

    def __init__(self):
        parameters = {}
        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(parameters=parameters,
                         component_obj=encoder,
                         random_state=0)
