from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "noisegate.tools",
        sources=["noisegate/tools.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name='noisegate',
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
)