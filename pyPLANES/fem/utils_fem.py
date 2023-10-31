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

from itertools import product
import numpy as np
import numpy.linalg as LA
from scipy.special import legendre

def dof_p_linear_system_master(_elem):
    if _elem.typ in [1, 8]:
        return np.array(_elem.dofs[3][:2] + _elem.dofs[3][2])
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[3][:3] + _elem.dofs[3][3] + _elem.dofs[3][4] +_elem.dofs[3][5])

def dof_p_linear_system_to_condense(_elem):
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[3][6])

def dof_up_linear_system_master(_elem):
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[0][:3] + _elem.dofs[0][3] + _elem.dofs[0][4] +_elem.dofs[0][5] + _elem.dofs[1][:3] + _elem.dofs[1][3] + _elem.dofs[1][4] +_elem.dofs[1][5]+
        _elem.dofs[3][:3] + _elem.dofs[3][3] + _elem.dofs[3][4] +_elem.dofs[3][5])

def dof_up_linear_system_to_condense(_elem):
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[0][6]+_elem.dofs[1][6]+_elem.dofs[3][6])

def dof_u_linear_system_master(_elem):
    if _elem.typ in [1, 8]:
        return np.array(_elem.dofs[0][:2] + _elem.dofs[0][3] + _elem.dofs[1][:2] + _elem.dofs[1][3])
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[0][:3] + _elem.dofs[0][3] + _elem.dofs[0][4] +_elem.dofs[0][5] + _elem.dofs[1][:3] + _elem.dofs[1][3] + _elem.dofs[1][4] +_elem.dofs[1][5])

def dof_ux_linear_system_master(_elem):
    if _elem.typ in [1, 8]:
        return np.array(_elem.dofs[0][:2] + _elem.dofs[0][2])
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[0][:3] + _elem.dofs[0][3] + _elem.dofs[0][4] +_elem.dofs[0][5])

def dof_uy_linear_system_master(_elem):
    if _elem.typ in [1, 8]:
        return np.array(_elem.dofs[1][:2] + _elem.dofs[1][2])
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[1][:3] + _elem.dofs[1][3] + _elem.dofs[1][4] +_elem.dofs[1][5])

def dof_u_linear_system_to_condense(_elem):
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[0][6]+_elem.dofs[1][6])

def dof_u_linear_system(_elem):
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[0][:3] + _elem.dofs[0][3] + _elem.dofs[0][4] +_elem.dofs[0][5] + _elem.dofs[0][6]+ _elem.dofs[1][:3] + _elem.dofs[1][3] + _elem.dofs[1][4] +_elem.dofs[1][5]+ _elem.dofs[1][6])

def dof_up_linear_system(_elem):
    if _elem.typ in [2, 9]:
        return np.array(_elem.dofs[0][:3] + _elem.dofs[0][3] + _elem.dofs[0][4] +_elem.dofs[0][5] + _elem.dofs[0][6]+ _elem.dofs[1][:3] + _elem.dofs[1][3] + _elem.dofs[1][4] +_elem.dofs[1][5]+ _elem.dofs[1][6]+
        _elem.dofs[3][:3] + _elem.dofs[3][3] + _elem.dofs[3][4] +_elem.dofs[3][5]+ _elem.dofs[3][6])

def dof_p_element(_elem):
    dof, orient = dof_element(_elem, 3)
    orient = np.diag(orient)
    elem_dof = local_dofs(_elem, "p")
    return dof, orient, elem_dof

def dof_u_element(_elem):
    dof_ux, orient_ux = dof_element(_elem, 0)
    dof_uy, orient_uy = dof_element(_elem, 1)
    elem_dof = local_dofs(_elem, "u")
    dof = dof_ux + dof_uy
    orient = np.diag(orient_ux + orient_uy)
    return dof, orient, elem_dof

def dof_ux_element(_elem):
    dof_ux, orient_ux = dof_element(_elem, 0)
    orient = np.diag(orient_ux)
    return dof_ux, orient

def dof_uy_element(_elem):
    dof_uy, orient_uy = dof_element(_elem, 1)
    orient = np.diag(orient_uy)
    return dof_uy, orient

def dof_up_element(_elem):
    dof_ux, orient_ux = dof_element(_elem, 0)
    dof_uy, orient_uy = dof_element(_elem, 1)
    dof_p, orient_p = dof_element(_elem, 3)
    elem_dof = local_dofs(_elem, "up")
    dof = dof_ux + dof_uy +dof_p
    orient = np.diag(orient_ux) # return a single one
    return dof, orient, elem_dof




def orient_element(_elem, f="p"):
    order = _elem.reference_element.order
    if _elem.typ in [2,9]: # Triangles
        # Orientation of the vertices
        orient = [1, 1, 1]
        # Orientation of the edges
        for _e, k in product(range(3), range(order-1)):
            orient.append(_elem.edges_orientation[_e]**k)
        # Orientation of the (unique) face
        orient += [1] * int((order-1)*(order-2)/2)
    elif _elem.typ in [1,8]: # Line Elements:
        orient = [1, 1]
        for _e, k in product(range(1), range(order-1)):
            orient.append(_elem.edges_orientation[_e]**k)
    if f == "u": # Duplication of the orientation for the two directions
        orient *= 2
    return np.diag(orient)

def local_dofs(_elem, field="p"):
    order = _elem.reference_element.order
    if _elem.typ in [2, 9]:
        # Local dofs
        nb_m = 3*order
        nb_c = int(((order-1)*(order-2))/2)
        nb_d = nb_m + nb_c
        if field == "p":
            elem_dof =dict()
            elem_dof["dof_m"] = slice(nb_m)
            elem_dof["dof_c"] = slice(nb_m, nb_m+nb_c)
        elif field == "u":
            elem_dof =dict()
            elem_dof["dof_m_x"] = slice(nb_m)
            elem_dof["dof_c_x"] = slice(nb_m, nb_d)
            elem_dof["dof_m_y"] = slice(nb_d, nb_d+nb_m)
            elem_dof["dof_c_y"] = slice(nb_d+nb_m, 2*nb_d)
        elif field == "up":
            elem_dof =dict()
            elem_dof["dof_m_x"] = slice(nb_m)
            elem_dof["dof_c_x"] = slice(nb_m, nb_d)
            elem_dof["dof_m_y"] = slice(nb_d, nb_d+nb_m)
            elem_dof["dof_c_y"] = slice(nb_d+nb_m, 2*nb_d)
            elem_dof["dof_m_p"] = slice(2*nb_d, 2*nb_d+nb_m)
            elem_dof["dof_c_p"] = slice(2*nb_d+nb_m, 3*nb_d)

    elif _elem.typ in [1, 8]:
        elem_dof = None
    return elem_dof

def normal_to_element(elem_1d, elem_2d):

    coord_e = elem_1d.coord

    n_ = coord_e[:, 1]- coord_e[:, 0]
    n_ = np.array([n_[1], -n_[0]])
    n_ = n_/LA.norm(n_)

    vec_2dto1d = elem_1d.get_center()-elem_2d.get_center()
    if np.dot(n_, vec_2dto1d)<0:
        n_ *= -1

    return n_


def dof_element(_elem, i_field):
    order = _elem.reference_element.order
    if _elem.typ in [2, 9]:
        # Only the master dofs (vertices and edges)
        # Pressure dofs of the vertices
        dof = [_elem.vertices[i].dofs[i_field] for i in range(3)]
        # Pressure dofs of the three edges
        dof = dof + _elem.edges[0].dofs[i_field] + _elem.edges[1].dofs[i_field] +_elem.edges[2].dofs[i_field]
        # Pressure dofs of the face functions
        # if _elem.faces[0].dofs != []:
        #     dof.extend(_elem.faces[0].dofs[i_field])
        # Orientation of the vertices
        orient = [1, 1, 1]
        # Orientation of the edges
        for _e, k in product(range(3), range(order-1)):
            orient.append(_elem.edges_orientation[_e]**k)
        # Orientation of the (unique) face
        orient += [1] * int((order-1)*(order-2)/2)
    elif _elem.typ in [1, 8]:
        # dof = dofs of the 2 vertices + of the edge
        dof = _elem.dofs[i_field][0:2]+_elem.dofs[i_field][2]
        orient = [1]*2 # Orientation of the vertices
        # Orientation of the edges
        orient.extend(_elem.edges_orientation[0]**np.arange(order-1))
        elem_dof = None
    return dof, orient

def create_legendre_table(n):
    out = np.zeros((n, n))
    for N in range(n):
        L = legendre(N)
        for J in range(n):
            out[N, J] = np.polyder(L, J)(1)
    return out