#!/usr/bin/env python

import glob
import os
import sys
from distutils.sysconfig import get_config_vars

# from setuptools import setup
from setuptools import setup

# Bail on Python < 3
assert sys.version_info[0] >= 3

# MF. This is a workaround to be able to build the library with MacOS
if sys.platform == 'darwin':
    vars = get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '-dynamiclib')
    os.environ["CC"] = "clang"

# NB. These are not really Python extensions (i.e., they do not
# Py_Initialize() and they do define main() ), we are just cheating to
# re-use the setuptools build support.

packages = ['rascil']
package_data = [i for p in packages for i in
                glob.glob(p + '/*/') + glob.glob(p + '/*/*/') + glob.glob(p + '/*/*/*/')]
setup(name='rascil',
      version='0.1.4',
      python_requires='>=3.6',
      description='Radio Astronomy Simulation, Calibration, and Imaging Library',
      long_description=open('README.md').read(),
      author='Tim Cornwell, Peter Wortmann, Bojan Nikolic, Feng Wang, Vlad Stolyarov',
      author_email='realtimcornwell@gmail.com',
      url='https://github.com/SKA-ScienceDataProcessor/rascil',
      license='Apache License Version 2.0',
      packages=(packages + package_data),
      test_suite="tests",
      tests_require=['pytest'],
      install_requires=['aotools',
                        'astropy>=4.0',
                        'bokeh',
                        'dask',
                        'distributed',
                        'h5py',
                        'jupyter',
                        'jupyter_contrib_nbextensions',
                        'matplotlib>3.1.3',
                        'numba',
                        'numpy',
                        'paramiko',
                        'photutils',
                        'python-casacore',
                        'graphviz',
                        'reproject',
                        'scikit-image',
                        'scipy',
                        'seqfile',
                        'ConfigParser',
                        'tabulate']
      )
