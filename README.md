# Flaxify

Tiny utility to make haiku models instantiation easier. It wraps `haiku.Module`s in a `hk.transform` without the need to add boiler plate lambda functions similiar to the behaviour of `flax.linen.Module`.

## Example usage

Standard haiku requires to create an extra function before a module can be transformed to pure functions:
```python
class MyHaikuModule(hk.Module):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, input):
        return hk.nets.MLP([128, self.output_size])(input)

hk.transform(hk.Linear(10))
# ValueError: All `hk.Module`s must be initialized inside an `hk.transform`.

def forward(input):
    return MyHaikuModule(10)

hk.transform(forward)
# Transformed(init=<function without_state>, apply=<function without_state>)
```

Adding the flaxify decorator, removes the need for the boiler plate:

```python
@flaxify
class MyHaikuModule(hk.Module):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, input):
        return hk.nets.MLP([128, self.output_size])(input)

MyHaikuModule
# Transformed(init=<function without_state>, apply=<function without_state>)
```

