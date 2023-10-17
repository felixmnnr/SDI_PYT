# setup.py
import os 
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
os.environ["CC"] = "gcc"

setup(
    ext_modules=cythonize("knn.pyx"),
    include_dirs=[np.get_include()]
)