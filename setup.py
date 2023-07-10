from Cython.Build import cythonize
from setuptools import setup
import numpy

setup(
    name='accelerated_simulation',
    ext_modules=cythonize('fast_cycle_loop.pyx',
                          compiler_directives={'language_level': "3",
                                               'c_string_type': 'unicode',
                                               'c_string_encoding': 'utf8',
                                               'boundscheck': False,
                                               'wraparound': False}),
    include_dirs=[numpy.get_include()],
    version='',
    url='',
    license='',
    author='Luis Schoppik',
    author_email='',
    description=''
)
