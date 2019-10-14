from .model_types import ModelTypes


def handle_model_types(model_type):
    """Handles model_type by either returning the ModelTypes or converting from a str

    Args:
        model_type (str or ModelTypes) : model type that needs to be handled

    Returns:
        ModelTypes
    """

    if isinstance(model_type, str):
        try:
            tpe = ModelTypes[model_type.upper()]
        except KeyError:
            raise KeyError('Model type \'{}\' does not exist'.format(model_type))
        return tpe
    if isinstance(model_type, ModelTypes):
        return model_type
    raise ValueError('`handle_model_types` was not passed a str or ModelTypes object')
