# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:52:50 2013

@author: René
"""

from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext

setup(
cmdclass = {"build_ext": build_ext},
ext_modules = [Extension("eosinterx",
                         ["eosinterx.pyx"],
					     include_dirs=[np.get_include()],
                         extra_compile_args=["-march=native", "-O3"]
                        ),
])
