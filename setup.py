from distutils.core import setup, Extension
import numpy

pydro = Extension(  'pydro',
                    sources=['pydro.c'],
                    include_dirs=[numpy.get_include(), '.'],
    )

setup ( name='pydro',
        version='0.1',
        description="Python reimplementation of Pedro Felzenszwalb's HoG features.",
        ext_modules=[pydro],
    )
