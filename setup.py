from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # cimport numpy を使うため

ext = Extension("calc", sources=["calc.pyx"], include_dirs=['.', get_include()])
setup(name="calc", ext_modules=cythonize([ext]))