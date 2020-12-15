#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# fem_classes.py
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
from pyPLANES.generic.elements_generic import GenericVertex, GenericEdge, GenericElement


class DgmEdge(GenericEdge):
    ''' TODO '''
    def __init__(self, tag, vertices, element):
        GenericEdge.__init__(self, tag, vertices, element)
        self.tag = tag
        self.vertices = vertices
        self.elements = []
        self.dofs = None
        self.sol = None

    def __str__(self):
        out = GenericEdge.__str__(self)
        # out += "dofs=" + format(self.dofs)+"\n"
        return out

class ImposedDisplacementDgmEdge(DgmEdge):
    def __init__(self, tag, vertices):
        DgmEdge.__init__(self, tag, vertices)


class RigidWallDgmEdge(DgmEdge):
    def __init__(self, tag, vertices):
        DgmEdge.__init__(self, tag, vertices)

class InternalFluidDgmEdge(DgmEdge):
    def __init__(self, tag, vertices):
        DgmEdge.__init__(self, tag, vertices)