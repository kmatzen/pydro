from intelccompiler import *
from distutils.core import setup, Extension
import numpy

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--compiler', default='gcc')
args, unknown = parser.parse_known_args()

if args.compiler == 'intelem':
    #MKLROOT = '/usr/local/intel/composer_xe_2013.5.192/mkl'
    MKLROOT = '/ld1/kmatzen/intel/mkl'

    libs = [
        'mkl_rt',
        'mkl_intel_lp64',
        'mkl_core',
        'mkl_intel_thread',
        'pthread', 
        'm',
        'iomp5',
    ]

    flags = [
        '-march=corei7',
        '-openmp',
        '-fp-model fast',
        '-wd1498'
    ]

    library_dirs = [
        MKLROOT+'/lib/intel64',
    ]

    include_dirs = [
        MKLROOT+'/include',
    ]

elif args.compiler == 'gcc':
    libs = [
        'cblas',
        'gomp',
    ]

    flags = [
        '-fopenmp',
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

    libraries=libs,

    extra_compile_args=[
        '-g', 
        '-O3', 
        '-Wall', 
        '-Werror', 
        '-Wno-long-long',
        '-funroll-loops',
    ]+flags+os.environ.get('CXXFLAGS','').split(),

    extra_link_args=os.environ.get('LDFLAGS','').split(),

    include_dirs=[
        numpy.get_include(), 
    ]+include_dirs,
)

pydro_train = Extension(
    'pydro._train',

    sources=[
        'src/pydro/_train.c'
    ],

    library_dirs=library_dirs,

    libraries=libs,

    extra_compile_args=[
        '-g', 
        '-O3', 
        '-Wall', 
        '-Werror', 
        '-Wno-long-long',
        '-funroll-loops',
    ]+flags+os.environ.get('CXXFLAGS','').split(),

    extra_link_args=os.environ.get('LDFLAGS','').split(),

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

    libraries=libs,

    extra_compile_args=[
        '-g', 
        '-O3', 
        '-Wall', 
        '-Werror', 
        '-Wno-long-long'
    ]+flags+os.environ.get('CXXFLAGS','').split(),

    extra_link_args=os.environ.get('LDFLAGS','').split(),

    include_dirs=[
        numpy.get_include(), 
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
        pydro_train,
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

