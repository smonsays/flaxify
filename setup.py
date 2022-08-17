from setuptools import find_namespace_packages, setup

setup(
    name="flaxify",
    version="0.0.1",
    description="Convert haiku modules to flax.",
    author="smonsays",
    url="https://github.com/smonsays/flaxify",
    license='MIT',
    install_requires=["dm_haiku"],
    packages=find_namespace_packages(),
)
