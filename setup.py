#!/usr/bin/env python

from distutils.core import setup

setup(
    name="lazytorch",
    version="0.1.1",
    description="Lazily-Initialized PyTorch Modules",
    author="Michael Malek",
    author_email="michaelgmalek@gmail.com",
    license="MIT",
    url="https://github.com/mgmalek/lazytorch",
    packages=["lazytorch", "lazytorch.layers"],
    install_requires=["torch"],
    python_requires=">=3.8",
)
