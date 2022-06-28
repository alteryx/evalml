from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types

class UnknownTypeDataCheck(DataCheck):
    """Check if there are a high number of features that are labelled as unknown by Woodwork."""

    def __init__(
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

    Examples:
        >>> import pandas as pd
        ...
        >>> 
    """
    messages = []

    X = infer_feature_types(X)
    row_unknowns = X.ww.select("unknown")
    if len(row_unknowns) / len(X.columns) >= self.unknown_percentage_threshold:
        print("Unknown")
        warning_msg = f"{len(row_unknowns)} out of {len(X)} rows are {self.pct_null_row_threshold*100}% or more of unknown type."

        messages.append(
            DataCheckWarning(
                message=warning_msg,
                data_check_name=self.name,
                message_code=DataCheckMessageCode.HIGH_NUMBER_OF_UNKNOWN_TYPE,
                details={
                    "rows": row_unknowns.index.tolist(),
                },
                action_options=[
                    DataCheckActionOption(
                        DataCheckActionCode.DROP_ROWS,
                        data_check_name=self.name,
                        metadata={"rows": row_unknowns.index.tolist()}
                    )
                ],
            ).to_dict()
        )
    return messages
