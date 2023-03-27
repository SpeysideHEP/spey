from typing import Text, Any, ClassVar


def get_function(
    class_base: ClassVar,
    function_name: Text,
    default: Any = None,
    exception: Exception = NotImplementedError,
    **kwargs,
) -> Any:
    """
    Function wrapper to be able to set a default value for the attributes that are not implemented

    :param class_base (`ClassVar`): Class basis where the function supposed to live
    :param function_name (`Text`): name of the function
    :param default (`Any`): default value that function needs to return.
    :param exception (`Exception`, default `NotImplementedError`): Exception that should be avoided.
    :param kwargs: function inputs
    :return `Any`: default value or the function output
    """
    try:
        output = getattr(class_base, function_name, default)
        if callable(output):
            output = output(**kwargs)
    except exception:
        output = default(**kwargs) if callable(default) else default

    return output
