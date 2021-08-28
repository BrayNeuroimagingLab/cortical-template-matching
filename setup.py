"""
Used to compile the Cython

@author Rylan Marianchuk
July 2021
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np


setup(
    ext_modules = cythonize("getRpthreaded.pyx", annotate=True, compiler_directives={'language_level':3}),
    include_dirs = [np.get_include()]
)
