import inspect

def get_function_defaults(fn):
    """get dict of name:default for a function's arguments"""
    s = inspect.signature(fn)
    return {k:v.default for k,v in s.parameters.items()}

def get_class_defaults(cls):
    """get the default argument values of a class constructor"""
    d = get_function_defaults(getattr(cls, '__init__'))
    # ignore `self` argument, insist on default values
    try:
        d.pop('self')
    except KeyError:
        raise ValueError("""
            no `self` argument found in class __init__
        """)
    assert [v is not inspect._empty for v in d.values()], """
            get_class_defaults should be used on constructors with keyword arguments only.
        """
    return d
