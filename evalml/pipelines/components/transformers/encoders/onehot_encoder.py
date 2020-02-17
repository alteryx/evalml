import category_encoders as ce

from .encoder import CategoricalEncoder


class OneHotEncoder(CategoricalEncoder):

    """Creates one-hot encoding for non-numeric data"""
    name = 'One Hot Encoder'
    _needs_fitting = True
    hyperparameter_ranges = {}

    def __init__(self):
        parameters = {}
        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(parameters=parameters,
                         component_obj=encoder,
                         random_state=0)
