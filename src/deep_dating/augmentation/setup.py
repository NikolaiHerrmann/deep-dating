
from distutils.core import setup
from Cython.Build import cythonize

setup(name="imagemorph_cython",
      ext_modules=cythonize("imagemorph_cython.pyx", compiler_directives={"language_level": "3"}))
