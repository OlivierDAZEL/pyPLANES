#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# utils_fem.py
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

from itertools import product
import numpy as np
from scipy.special import legendre



def dof_p_element(_elem):
    dof, orient = dof_element(_elem, 3)
    return dof, orient


def dof_u_element(_elem):
    dof_ux, orient_ux = dof_element(_elem, 0)
    dof_uy, orient_uy = dof_element(_elem, 1)
    dof_uz, orient_uz = dof_element(_elem, 2)
    dof = dof_ux + dof_uy + dof_uz
    orient = orient_ux + orient_uy + orient_uz
    return dof, orient


def dof_element(_elem, i_field):
    order = _elem.reference_element.order
    if _elem.typ == 2:
        # Pressure dofs of the vertices
        dof = [_elem.vertices[i].dofs[i_field] for i in range(3)]
        # Pressure dofs of the three edges
        dof = dof + _elem.edges[0].dofs[i_field] + _elem.edges[1].dofs[i_field] +_elem.edges[2].dofs[i_field]
        # Pressure dofs of the face functions
        if _elem.faces[0].dofs != []:
            dof.extend(_elem.faces[0].dofs[i_field])
        # Orientation of the vertices
        orient = [1, 1, 1]
        # Orientation of the edges
        for _e, k in product(range(3), range(order-1)):
            orient.append(_elem.edges_orientation[_e]**k)
        # Orientation of the (unique) face
        orient += [1] * int((order-1)*(order-2)/2)
    elif _elem.typ == 1:
        # dof = dofs of the 2 vertices + of the edge
        dof = _elem.dofs[i_field][0:2]+_elem.dofs[i_field][2]
        orient = [1]*2 # Orientation of the vertices
        # Orientation of the edges
        orient.extend(_elem.edges_orientation[0]**np.arange(order-1))
    return dof, orient

def create_legendre_table(n):
    out = np.zeros((n, n))
    for N in range(n):
        L = legendre(N)
        for J in range(n):
            out[N, J] = np.polyder(L, J)(1)
    return out