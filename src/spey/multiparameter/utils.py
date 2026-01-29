import inspect


def inspect_function(fn):
    """
    Returns information about a function's arguments.
    """
    sig = inspect.signature(fn)

    args = []
    kwargs = []
    has_varargs = False
    has_varkw = False

    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            has_varkw = True
        elif param.default is inspect.Parameter.empty:
            args.append(name)
        else:
            kwargs.append((name, param.default))

    return {
        "n_args": len(args),
        "args": args,
        "kwargs": dict(kwargs),
        "has_varargs": has_varargs,
        "has_varkw": has_varkw,
    }
