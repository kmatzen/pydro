from distutils.core import setup, Extension
import numpy

from intelccompiler import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--compiler', default='gcc')
args, unknown = parser.parse_known_args()

if args.compiler == 'intelem':
    libs = [
        'mkl_rt', 
        'mkl_intel_ilp64', 
        'mkl_sequential', 
        'mkl_core',
        'iomp5',
    ]

    flags = [
        '-DMKL_ILP64',
        '-fp-model fast',
    ]

    library_dirs = [
        '/usr/local/intel/composer_xe_2013.5.192/mkl/lib/intel64',
        '/usr/local/intel/lib/intel64',
    ]

    include_dirs = [
        '/usr/local/intel/composer_xe_2013.5.192/mkl/include',
    ]

elif args.compiler == 'gcc':
    libs = [
        'cblas',
        'gomp',
    ]

    flags = [
        '-ffast-math',
    ]

    library_dirs = [
        '/usr/lib/atlas-base',
    ]

    include_dirs = [
        '/usr/include/atlas',
    ]

else:
    raise Exception ('intelem and gcc are the only supported compilers.')

pydro_detection = Extension(
    'pydro._detection',

    sources=[
        'src/pydro/_detection.c'
    ],

    library_dirs=library_dirs,

    libraries=[
        'dl', 
        'pthread', 
        'm', 
    ]+libs,

    extra_compile_args=[
        '-fopenmp', 
        '-g', 
        '-m64', 
        '-O3', 
        '-Wall', 
        '-Werror', 
        '-Wno-long-long',
        '-funroll-loops',
    ]+flags,

    include_dirs=[
        numpy.get_include(), 
    ]+include_dirs,
)

pydro_features = Extension(
    'pydro._features',

    sources=[
        'src/pydro/_features.c'
    ],

    library_dirs=library_dirs,

    libraries=[
        'dl', 
        'pthread', 
        'm', 
    ]+libs,

    extra_compile_args=[
        '-fopenmp', 
        '-g', 
        '-m64', 
        '-O3', 
        '-Wall', 
        '-Werror', 
        '-Wno-long-long'
    ]+flags,

    include_dirs=[
        numpy.get_include(), 
        '.'
    ]+include_dirs,
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

