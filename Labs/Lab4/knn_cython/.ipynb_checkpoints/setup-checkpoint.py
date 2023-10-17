from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("knn_cython.pyx"),
    include_dirs=[np.get_include()]  # This is to include NumPy headers
)