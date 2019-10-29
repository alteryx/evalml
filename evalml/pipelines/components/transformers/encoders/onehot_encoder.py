import category_encoders as ce

from .encoder import CategoricalEncoder

from evalml.pipelines.components import ComponentTypes


class OneHotEncoder(CategoricalEncoder):

    """Creates one-hot encoding for non-numeric data"""
    name = 'One Hot Encoder'
    component_type = ComponentTypes.CATEGORICAL_ENCODER
    hyperparameter_ranges = {}

    def __init__(self):
        parameters = {}
        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(name=self.name,
                         component_type=self.component_type,
                         parameters=parameters,
                         component_obj=encoder,
                         needs_fitting=True,
                         random_state=0)
