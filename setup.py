#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

#
# To develop:  "python setup.py develop"
# To build (pre-install): "python setup.py build"
# To install: "python setup.py install"

import io
import os
import sys
from numpy import get_include as np_get_include
from shutil import rmtree

from setuptools import find_packages, setup, Command
from setuptools.extension import Extension
from Cython.Build import cythonize

# Package meta-data.
NAME = 'NoiseGate'
DESCRIPTION = 'Denoising of images and image sequences'
URL = 'https://github.com/drzowie/image-noise-gate'
EMAIL = 'deforest@boulder.swri.edu'
AUTHOR = 'Craig DeForest'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.1'

# What packages are required for this module to be executed?
REQUIRED = [
    'pytest',     # testing
    'numpy',      # basic numerics
    'cython',     # static compilation and C objects
]

# What packages are optional?
EXTRAS = {
        'docs':     'jupyter',    # to read the ipynb doc file
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Set up extensions - trivial with just helpers, but 
# useful later if we have to link in C libraries etc.
extensions = [
    Extension("noisegate.tools",
              [here+"/noisegate/tools.pyx"],
              include_dirs = [np_get_include()],
              )
    ]

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where='noisegate'),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    requires=REQUIRED,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='LGPL-v3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    zip_safe = False,
    ext_modules = cythonize(extensions),
)
