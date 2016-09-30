#!/usr/bin/env python

"""Setup script for Apriori."""

# For full authorship and copyright information, see the mit-license file
__author__ = 'Aaron Hosford'

from setuptools import setup
from os import path

from apriori import __version__

PACKAGE_NAME = 'apriori'

here = path.abspath(path.dirname(__file__))

long_description = """
Implements the Apriori algorithm for itemset and association rule learning.
See http://en.wikipedia.org/wiki/Apriori_algorithm for full a description.
"""

setup(
    name='apriori',
    version=__version__,
    description='Apriori Algorithm (association rule learning)',
    long_description=long_description,
    url='https://github.com/SaintAttila/Apriori',
    author=__author__,
    author_email='hosford42@gmail.com',
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='apriori algorithm itemsets associative rule learning',
    py_modules=['apriori'],
)
