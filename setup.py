from distutils.core import setup, Extension
import numpy

use_mkl = False
if use_mkl:
    blas_libs = [
        'mkl_rt', 
        'mkl_intel_ilp64', 
        'mkl_gnu_thread', 
        'mkl_core'
    ]
    blas_flags = [
        '-D__USE_MKL__', 
        '-DMKL_ILP64'
    ]
    blas_library_dirs = [
        '/usr/local/intel/composer_xe_2013.5.192/mkl/lib/intel64',
    ]
    blas_include_dirs = [
        '/usr/local/intel/composer_xe_2013.5.192/mkl/include',
    ]
else:
    blas_libs = ['cblas']
    blas_flags = []
    blas_library_dirs = [
        '/usr/lib/atlas-base'
    ]
    blas_include_dirs = [
        '/usr/include/atlas'
    ]

pydro_detection = Extension(
    'pydro._detection',

    sources=[
        'src/pydro/_detection.c'
    ],

    library_dirs=blas_library_dirs,

    libraries=[
        'dl', 
        'pthread', 
        'm', 
        'gomp'
    ]+blas_libs,

    extra_compile_args=[
        '-fopenmp', 
        '-g', 
        '-m64', 
        '-O3', 
        '-Wall', 
        '-Werror', 
        '-Wno-long-long',
        '-funroll-loops',
    ]+blas_flags,

    include_dirs=[
        numpy.get_include(), 
    ]+blas_include_dirs,
)

pydro_features = Extension(
    'pydro._features',
    sources=['src/pydro/_features.c'],
    libraries=['dl', 'pthread', 'm', 'gomp'],
    extra_compile_args=['-fopenmp', '-g', '-m64', '-O3', '-Wall', '-Werror', '-Wno-long-long'],
    include_dirs=[numpy.get_include(), '.'],
)

setup ( 
    name='pydro',
    version='0.0',
    description="Python reimplementation of Pedro Felzenszwalb's HoG features.",
    author='Kevin Matzen',
    author_email='kmatzen@cs.cornell.edu',
    url='https://github.com/kmatzen/pydro',
    download_url='https://github.com/kmatzen/pydro/tarball/master',
    keywords=['dpm', 'vision', 'recognition', 'detection', 'deformable', 'parts', 'model'],
    classifiers=[],
    ext_modules=[
        pydro_detection,
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

