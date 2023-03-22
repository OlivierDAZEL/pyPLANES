#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# utils_fem.py
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

import numpy as np
import numpy.linalg as LA
from scipy.linalg import expm
from numpy import sqrt

def chebyshev(x, N, D=0):
    Tn = np.zeros((x.size, N, D+1))
    Tn[:, 0, 0] = 1.0
    Tn[:, 1, 0] = x
    if D > 0:
        Tn[:, 1, 1] = 1.0
    for n in range(2, N):
        Tn[:, n, 0] = 2*x*Tn[:, n-1, 0] - Tn[:, n-2, 0]
        for d in range(1, D+1):
            Tn[:, n, d] = 2*x*Tn[:, n-1, d] + 2 * \
                d*Tn[:, n-1, d-1] - Tn[:, n-2, d]
    return(Tn)

def chebyshev_nodes(M):
    m = np.arange(M)
    xi = np.cos((2*m+1)/M*np.pi/2)
    return(xi)