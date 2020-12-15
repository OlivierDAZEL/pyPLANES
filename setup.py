#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# setup.py
#
# This file is part of pymls, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2020
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@kth.se>
# 	Peter Göransson <pege@kth.se>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#

from setuptools import setup, find_packages


with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(name='pyPLANES',
    version='0.2.0',
    description='python Porous LAum NumErical Simulator',
    long_description=long_description,
    url='https://github.com/OlivierDAZEL/pyplanes',
    author='O. Dazel',
    author_email='olivier.dazel@univ-lemans.fr',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy>=1.13.0',
        'PyYAML>=5.1',
        'matplotlib>=2.0',
        'quadpy',
        'termcolor',
        'scipy',
        'mediapack>=0.4.0',
        'pymls>=1.7.0',
        'pyvtk'
    ],
)
