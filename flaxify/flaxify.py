import haiku as hk
from haiku._src.transform import Transformed


def flaxify(m: hk.Module, transform_fn=hk.transform_with_state) -> Transformed:
    """
    Wraps a haiku.Module in a transformed forward function making it behave
    similar to a flax.nn.Module.

    Args:
        m: the haiku Module to be wrapped
        transform_fn: the haiku transofrm to apply (either transform or transform_with_state)

    Example:
        >>> # Instead of having to define a boiler plate lambda function
        >>> my_linear = hk.transform(lambda x: hk.Linear(10)(x))
        >>> # Flaxified modules are automatically converted to pure functions after module instantiation
        >>> FlaxifiedLinear = flaxify(hk.Linear)
        >>> my_linear = FlaxifiedLinear(10)
    """
    def fun(*haiku_module_args, **haiku_module_kwargs):
        def forward(*forward_args, **forward_kwargs):
            return m(*haiku_module_args, **haiku_module_kwargs)(*forward_args, **forward_kwargs)

        return transform_fn(forward)

    return fun
