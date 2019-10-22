import category_encoders as ce

from .encoder import Encoder

from evalml.pipelines.components import ComponentTypes


class OneHotEncoder(Encoder):

    """Creates one-hot encoding for non-numeric data"""
    hyperparameter_ranges = {}

    def __init__(self):
        name = 'One Hot Encoder'
        component_type = ComponentTypes.ENCODER
        parameters = {}
        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(name=name,
                         component_type=component_type,
                         parameters=parameters,
                         component_obj=encoder,
                         needs_fitting=True,
                         random_state=0)
