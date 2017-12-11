#!/usr/bin/env python
'''
Setup script for the mvacfg package
'''

__author__ = 'Miguel Ramos Pernas'
__email__  = 'miguel.ramos.pernas@cern.ch'


# Python
import os
from setuptools import setup, find_packages

#
# Version of the package. Before a new release is made
# just the "version_info" must be changed.
#
version_info = (0, 0, 0)
version = '.'.join(map(str, version_info))

# Setup function
setup( 
    
    name = 'mvacfg',
    
    version = version,
    
    description = 'Package to track MVA configurations using scikit-learn',
    
    # Read the long description from the README
    long_description = open('README.rst').readlines(),
    
    # Keywords to search for the package
    keywords = 'mva bdt hep',

    # Find all the packages in this directory
    packages = find_packages(),

    # Install scripts
    scripts = ['scripts/read_mva_configs'],

    # Requisites
    install_requires = ['configparser', 'matplotlib', 'numpy', 'pytest', 'scikit-learn'],

    # Test requirements
    setup_requires = ['pytest-runner'],
    
    tests_require = ['pytest'],
    )


# Create a module with the versions
version_file = open('mvacfg/version.py', 'wt')
version_file.write("""\
'''
Auto-generated module holding the version of the mvacfg package
'''

__version__ = "{}"
__version_info__ = {}

__all__ = ['__version__', '__version_info__']
""".format(version, version_info))
version_file.close()
