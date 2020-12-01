#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# entity_classes.py
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

import numpy as np
import numpy.linalg as LA
from numpy import pi
from itertools import chain

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

from mediapack import Air


class GmshEntity():
    def __init__(self, **kwargs):
        self.dim = kwargs["dim"]
        self.tag = kwargs["tag"]
        self.physical_tags = kwargs["physical_tags"]
        entities = kwargs["entities"]
        if "condition" not in list(self.physical_tags.keys()):
            self.physical_tags["condition"] = None
        if "model" not in list(self.physical_tags.keys()):
            self.physical_tags["model"] = None
        if self.dim == 0:
            self.neighbouring_curves = []
            self.x = kwargs["x"]
            self.y = kwargs["y"]
            self.z = kwargs["z"]
            self.coord = np.array([self.x, self.y, self.z])
            self.neighbours = []
        elif self.dim == 1:
            self.neighbouring_surfaces = []
            self.bounding_points = [next((e for e in entities if e.tag == abs(t)), None) for t in kwargs["bounding_points"]]
            self.center = np.array([0., 0., 0.])
            for p in self.bounding_points:
                self.center += [p.x, p.y, p.z]
            self.center /= len(self.bounding_points)
            for _e in self.bounding_points:
                _e.neighbouring_curves.append(self)
        elif self.dim == 2:
            self.neighbouring_surfaces = [] # Neighbouring 2D entities, will be completed in preprocess
            self.bounding_curves = [next((e for e in entities if e.tag == abs(t)), None) for t in kwargs["bounding_curves"]]
            for _e in self.bounding_curves:
                _e.neighbouring_surfaces.append(self)
            self.center = np.array([0., 0., 0.])
            for c in self.bounding_curves:
                self.center += c.center
            self.center /= len(self.bounding_curves)


    def __str__(self):
        out = "Entity / tag={} / dim= {}\n".format(self.tag, self.dim)
        out += "Physical tags={}\n".format(self.physical_tags)
        if self.dim == 0:
            out += "Belongs to curves "
            for _c in self.neighbouring_curves:
                out += "{} ({}) ".format(_c.tag, _c.physical_tags["condition"])
            out += "\n"
        if self.dim == 1:
            out += "Related points "
            for _b in self.bounding_points:
                out += "{} ({}) ".format(_b.tag,_b.physical_tags["condition"])
            out += "\n"
        if self.dim == 2:
            out += "Related curves "
            for _c in self.bounding_curves:
                out += "{} ({}) ".format(_c.tag,_c.physical_tags["condition"])
            out += "\n"
        return out

class FemEntity(GmshEntity):
    def __init__(self, **kwargs):
        GmshEntity.__init__(self, **kwargs)
        # self.order = kwargs["order"]
        self.elements = []
    def __str__(self):
        # out = GmshEntity.__str__(self)
        out = "Fem" + GmshEntity.__str__(self)
        # out += "order:{}\n".format(self.order)
        # related_elements = [_el.tag for _el in self.elements]
        # out  += "related elements={}\n".format(related_elements)
        return out

    def condensation(self, omega):
        return [], [], [], [], [], []

    def update_frequency(self, omega):
        pass


    def elementary_matrices(self, _elem):
        """"  Create elementary matrices of the entity"""
        pass

    def update_system(self, omega):
        """
        Loop on the element of the entities

        Parameters
        ----------
        omega : real or complex
            circular frequency 

        Returns
        ----------
        A_i, A_j, A_v, T_i, T_j, T_v, F_i, F_v

        """
        raise NameError("update_system should be implemented in children classes") 

    def link_elem(self,n):
        self.elements.append(n)



