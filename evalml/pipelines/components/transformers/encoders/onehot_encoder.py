import category_encoders as ce

from .encoder import CategoricalEncoder


class OneHotEncoder(CategoricalEncoder):

    """Creates one-hot encoding for non-numeric data"""
    name = 'One Hot Encoder'
    hyperparameter_ranges = {}

    def __init__(self, parameters={}, component_obj=None, random_state=0):
        assert component_obj is None, "Cannot provide component_obj to this component"
        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(parameters=parameters,
                         component_obj=encoder,
                         random_state=random_state)
