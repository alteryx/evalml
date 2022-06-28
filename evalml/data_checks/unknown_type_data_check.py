from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)


class UnknownTypeDataCheck(DataCheck):
    """Check if there are a high number of features that are labelled as unknown by Woodwork."""

    def _init_(
        self,
        unknown_percentage_threshold=0.50,
    ):
        if not 0 <= unknown_percentage_threshold <= 1:
            raise ValueError(
                "`unknown_percentage_threshold` must be a float between 0 and 1, inclusive."
            )
        self.unknown_percentage_threshold = unknown_percentage_threshold


def validate(self, X, y=None):
    """Check if there are any rows or columns that have a high percentage of unknown types.

    Args:
        X (pd.DataFrame, np.ndarray): Features.

    Returns:
        dict: A dictionary with warnings if any columns
    """
    messages = []

    X = infer_feature_types(X)
