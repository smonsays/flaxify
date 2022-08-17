# Flaxify

Tiny utility to simplify the instantiation of [haiku](https://github.com/deepmind/dm-haiku) models. The `@flaxify` decorator automatically wraps a `haiku.Module`s in a `hk.transform` without the need to add boiler plate lambda functions similiar to the behaviour of [flax](https://github.com/google/flax).

## Example usage

`haiku` requires creating an extra function before a module can be transformed to pure functions:
```python
class MyHaikuModule(hk.Module):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, input):
        return hk.nets.MLP([128, self.output_size])(input)

hk.transform(MyHaikuModule(10))
# ValueError: All `hk.Module`s must be initialized inside an `hk.transform`.

def forward(input):
    return MyHaikuModule(10)

hk.transform(forward)
# Transformed(init=<function without_state>, apply=<function without_state>)
```

Adding the flaxify decorator, removes the boiler plate code:
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

## Installation

Install `flaxify` using pip:
```
pip install git+https://github.com/smonsays/flaxify
```
or simply copy [`flaxify.py `](https://github.com/smonsays/flaxify/blob/main/flaxify/flaxify.py).
