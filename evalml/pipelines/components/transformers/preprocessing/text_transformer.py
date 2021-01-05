from evalml.pipelines.components.transformers import Transformer
from evalml.utils.logger import get_logger

logger = get_logger(__file__)


class TextTransformer(Transformer):
    """Base class for all transformers working with text features"""

    def __init__(self, component_obj=None, random_state=0, **kwargs):
        """Creates a transformer to perform TF-IDF transformation and Singular Value Decomposition for text columns.

        Arguments:
            random_state (int, np.random.RandomState): Seed for the random number generator.
        """
        parameters = {}
        parameters.update(kwargs)

        super().__init__(parameters=parameters,
                         component_obj=component_obj,
                         random_state=random_state)

    def _get_text_columns(self, X):
        """Returns the ordered list of columns names in the input which have been designated as text columns."""
        text_column_vals = X.select('natural_language')
        text_columns = list(text_column_vals.to_dataframe().columns)
        return text_columns
