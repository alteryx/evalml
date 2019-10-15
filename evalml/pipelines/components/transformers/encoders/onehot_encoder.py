import category_encoders as ce

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.transformers import Transformer


class OneHotEncoder(Transformer):

    """Creates one-hot encoding for non-numeric data"""
    hyperparameters = {}

    def __init__(self):
        self.name = 'One Hot Encoder'
        self.component_type = ComponentTypes.ENCODER
        self.parameters = {}

        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(name=self.name, component_type=self.component_type, parameters=self.parameters, needs_fitting=True, component_obj=encoder)
