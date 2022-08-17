"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
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
