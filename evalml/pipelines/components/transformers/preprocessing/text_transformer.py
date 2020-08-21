from evalml.pipelines.components.transformers import Transformer
from evalml.utils.logger import get_logger

logger = get_logger(__file__)


class TextTransformer(Transformer):
    """Base class for all transformers working with text features"""

    def __init__(self, text_columns=None, component_obj=None, random_state=0, **kwargs):
        """Creates a transformer to perform TF-IDF transformation and Singular Value Decomposition for text columns.

        Arguments:
            text_columns (list): list of feature names which should be treated as text features.
            random_state (int, np.random.RandomState): Seed for the random number generator.
        """
        parameters = {'text_columns': text_columns}
        parameters.update(kwargs)

        self._all_text_columns = text_columns or []
        super().__init__(parameters=parameters,
                         component_obj=component_obj,
                         random_state=random_state)

    def _get_text_columns(self, X):
        """Returns the ordered list of columns names in the input which have been designated as text columns."""
        columns = []
        missing_columns = []
        for col_name in self._all_text_columns:
            if col_name in X.columns:
                columns.append(col_name)
            else:
                missing_columns.append(col_name)
        if len(columns) == 0:
            raise AttributeError("None of the provided text column names match the columns in the given DataFrame")
        if len(columns) < len(self._all_text_columns):
            logger.warn("Columns {} were not found in the given DataFrame, ignoring".format(missing_columns))
        return columns
