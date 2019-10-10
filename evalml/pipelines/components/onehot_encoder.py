import category_encoders as ce

from .component_types import ComponentTypes
from .transformer import Transformer


class OneHotEncoder(Transformer):

    """Creates one-hot encoding for non-numeric data"""

    def __init__(self):
        self.name = 'One Hot Encoder'
        self.component_type = ComponentTypes.ENCODER
        self.hyperparameters = {}
        self.parameters = {}
        encoder = ce.OneHotEncoder(use_cat_names=True, return_df=True)
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, parameters=self.parameters, needs_fitting=True, component_obj=encoder)
