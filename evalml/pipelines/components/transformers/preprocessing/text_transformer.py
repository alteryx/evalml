from evalml.pipelines.components.transformers import Transformer
from evalml.utils import _put_into_original_order
from evalml.utils.logger import get_logger

logger = get_logger(__file__)


class TextTransformer(Transformer):
    """Base class for all transformers working with text features.

    Arguments:
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def __init__(self, component_obj=None, random_seed=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)

        super().__init__(
            parameters=parameters, component_obj=component_obj, random_seed=random_seed
        )

    def _get_text_columns(self, X):
        """Returns the ordered list of columns names in the input which have been designated as text columns."""
        return _put_into_original_order(
            X, list(X.ww.select("NaturalLanguage", return_schema=True).columns)
        )
