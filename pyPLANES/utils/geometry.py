#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# utils_geometry.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
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

import numpy.linalg as LA

def getOverlap(a, b):
    return max(0, min(max(a), max(b)) - max(min(a), min(b)))

def local_abscissa(p_0, p_1, p_c):
    n = LA.norm(p_1 - p_0)
    return (p_c-p_0).dot(p_1 - p_0)/n**2