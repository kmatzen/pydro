from distutils.core import setup, Extension
import numpy

pydro_detect = Extension(
    'pydro._detect',
    sources=['src/pydro/_detect.c'],
    library_dirs=['/usr/local/intel/composer_xe_2013.5.192/mkl/lib/intel64'],
    libraries=['mkl_rt', 'mkl_intel_ilp64', 'mkl_gnu_thread', 'mkl_core', 'dl', 'pthread', 'm', 'gomp'],
    extra_compile_args=['-fopenmp', '-g', '-DMKL_ILP64', '-m64', '-O3', '-Wall', '-Werror', '-Wno-long-long'],
    include_dirs=[numpy.get_include(), '.', '/usr/local/intel/composer_xe_2013.5.192/mkl/include'],
)

pydro_features = Extension(
    'pydro._features',
    sources=['src/pydro/_features.c'],
    library_dirs=['/usr/local/intel/composer_xe_2013.5.192/mkl/lib/intel64'],
    libraries=['mkl_rt', 'mkl_intel_ilp64', 'mkl_gnu_thread', 'mkl_core', 'dl', 'pthread', 'm', 'gomp'],
    extra_compile_args=['-fopenmp', '-g', '-DMKL_ILP64', '-m64', '-O3', '-Wall', '-Werror', '-Wno-long-long'],
    include_dirs=[numpy.get_include(), '.', '/usr/local/intel/composer_xe_2013.5.192/mkl/include'],
)

setup ( 
    name='pydro',
    version='0.1',
    description="Python reimplementation of Pedro Felzenszwalb's HoG features.",
    ext_modules=[
        pydro_detect,
        pydro_features,
    ],
    packages=['pydro'],
    package_dir={'pydro':'src/pydro'},

    scripts=[
        'scripts/voc-dpm2pydro',
    ],
    requires=[
        'msgpack',
        'numpy',
        'scipy',
    ],
)

