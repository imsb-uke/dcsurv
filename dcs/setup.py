#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(name='dcs',
      version='0.0.1',
      description='Discrete Calibrated Survival',
      author='Patrick Fuhlert',
      author_email='patrick.fuhlert.work@gmail.com',
      url='https://gitlab.com/pfuhlert/discrete-calibrated-survival/',
      packages=find_packages(),
      include_package_data=True,
      )
