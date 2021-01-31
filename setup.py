#!/usr/bin/env python
import glob
import os

from setuptools import find_packages
from setuptools import setup

setup(
    name="deepv2d",
    version="0.1",
    author="chenlinghao",
    packages=find_packages(exclude=("configs", "tests",)),
    # ext_modules=get_extensions(),
    cmdclass={},
)
