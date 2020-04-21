#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# import_msh_file.py
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

# from pymls.media import Air
from pymls import from_yaml



from pyPLANES.fem.elements.reference_elements import Ka, Kt
from pyPLANES.classes.fem_classes import Vertex, Element
from pyPLANES.classes.entity_classes import GmshEntity, FemEntity, IncidentPwFem, PeriodicityFem, RigidWallFem, Pem98Fem, \
        AirFem, TransmissionPwFem



def load_msh_file(self, p):
    print("Reading "+ p.name_project + ".msh")
    f = open(p.name_project + ".msh", "r")
    _f = "_"
    while _f:
        _f = f.readline().strip()
        if _f.startswith('$'):
            tag = _f[1:]
            if tag == "MeshFormat":
                self.MeshFormat = f.readline()
            if tag == "PhysicalNames":
                physical_names(self, f)
            if tag == "Entities":
                entities(self, f, p)
            if tag == "PartitionedEntities":
                partition(self, f)
            if tag == "Nodes":
                nodes(self, f)
            if tag == "Elements":
                elements(self, f, p)
            if tag == "Periodic":
                periodic(self, f)
            _ = f.readline()
            if _.strip() != "$End"+ tag:
                raise NameError("Error in GMSH file importation at tag:" +tag)


def dict_physical_tags(self, _list):
    ''' create a dict from gmsh file physical tags.
        Possible keys: model and materials'''
    d = [self.physical_names[int(_)] for _ in _list]
    d = [_.split("=") for _ in d]
    key_list = [_[0] for _ in d]
    value_list = [_[1] for _ in d]
    d = dict(zip(key_list, value_list))
    return d

def physical_names(self, f):
    ''' Importation of Physical names'''
    num_physical_names = int(f.readline())
    self.physical_names = dict()
    for __ in range(num_physical_names):
        _ = f.readline().split()
        tag = int(_[1])
        key = " ".join(_[2:])[1:-1]
        self.physical_names[tag] = key

def entities(self, f, p):
    ''' creation of the list of entities '''
    _p, num_curves, num_surfaces, num_volumes = readl_int(f)
    for __ in range(_p):
        _f = f.readline().split()
        # print(f)
        tag = int(_f[0])
        # print(tag)
        x, y, z = float(_f[1]), float(_f[2]), float(_f[3])
        # print(x)
        # print(y)
        # print(z)
        num_physical_tags = int(_f[4])
        physical_tags = dict_physical_tags(self, _f[8:8+num_physical_tags])
        _ = GmshEntity(dim=0, tag=tag, physical_tags=physical_tags, x=x, y=y, z=z)
        self.entities.append(_)
    for _icurve in range(num_curves):
        _f = f.readline().split()
        tag = int(_f[0])
        num_physical_tags = int(_f[7])
        physical_tags = dict_physical_tags(self, _f[8:8+num_physical_tags])
        _ = 8+num_physical_tags
        num_bounding_points = int(_f[_])
        bounding_points =[int(_l) for _l in _f[_+1:]] if num_bounding_points != 0 else []
        if "model" in physical_tags.keys():
            if physical_tags["model"] == "FEM1D":
                if physical_tags["condition"] == "Incident_PW":
                    _ = IncidentPwFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, p=p)
                    self.model_entities.append(_)
                elif physical_tags["condition"] == "Transmission":
                    _ = TransmissionPwFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, p=p)
                    self.model_entities.append(_)
                elif physical_tags["condition"] == "Rigid Wall" :
                    _ = RigidWallFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, p=p)
                elif physical_tags["condition"] == "Periodicity" :
                    _ = PeriodicityFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, p=p)
                else:
                    raise NameError("FEM1D entity without physical condition")
        else: # No numerical model
            _ = GmshEntity(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points)
        self.entities.append(_)
    for _surf in range(num_surfaces):
        _f = f.readline().split()
        tag = int(_f[0])
        num_physical_tags = int(_f[7])
        physical_tags = dict_physical_tags(self,_f[8:8+num_physical_tags])
        _ = 8+num_physical_tags
        num_bounding_curves = int(_f[_])
        bounding_curves = [int(_l) for _l in _f[_+1:]] if num_bounding_curves != 0 else []
        if "model" in physical_tags.keys():
            if physical_tags["model"].startswith("FEM"):
                if "mat" in physical_tags.keys():
                    if physical_tags["mat"].split()[0] == "Air":
                        _ = AirFem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, p=p)
                        self.model_entities.append(_)
                    else:
                        mat = from_yaml(self.materials_directory + physical_tags["mat"].split()[0] +".yaml")
                        if mat.MODEL == "eqf":
                            _ = EqfFem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, p=p, mat=mat)
                            self.model_entities.append(_)
                        elif mat.MODEL == "pem":
                            if physical_tags["model"] == "FEM01":
                                _ = Pem01Fem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, p=p, mat=mat)
                                self.model_entities.append(_)
                            else:
                                _ = Pem98Fem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, p=p, mat=mat)
                                self.model_entities.append(_)
        self.entities.append(_)
    for __ in range(num_volumes):
        pass
    # ######
    _ = [_ent.tag for _ent in self.entities]
    self.entity_tag=[None]*(max(_)+1)
    for i,index in enumerate(_):
        self.entity_tag[index] = i

def partition(self, f):
    self.PartitionedEntities = []
    num_partitions = int(f.readline())
    num_ghostEntities = int(f.readline())
    num_points, num_curves, num_surfaces, num_volumes = readl_int(f)
    # print("Partitions")
    # print("_p, num_curves, num_surfaces, num_volumes={},{},{},{}".format(_p, num_curves, num_surfaces, num_volumes))
    for _ipoint in range(_p):
        _f = f.readline().split()
        tag = int(_f[0])
        parent_dim = int(_f[1])
        parent_tag = int(_f[2])
        num_partitions = int(_f[3])
        partitionTag =[int(l) for l in _f[4:4+num_partitions]]
        _ = 4+num_partitions
        x, y, z = float(_f[_]), float(_f[_+1]), float(_f[_+2])
        num_physical_tags = int(_f[_+3])
        _ += 4
        physical_tags = dict_physicalTag(self,_f[_:_+num_physical_tags])
    for _icurve in range(num_curves):
        _f = f.readline().split()
        tag = int(_f[0])
        parent_dim = int(_f[1])
        parent_tag = int(_f[2])
        num_partitions = int(_f[3])
        partitionTag =[int(l) for l in _f[4:4+num_partitions]]
        _ = 4+num_partitions+6 # to skip min and max
        num_physical_tags = int(_f[_])
        _ += 1
        physical_tags = dict_physicalTag(self,_f[_:_+num_physical_tags])
        _ = _+num_physical_tags
        num_bounding_points = int(_f[_])
        _ += 1
        bounding_points = [int(l) for l in _f[_:_+num_bounding_points]]
    for _isurf in range(num_surfaces):
        _f = f.readline().split()
        tag = int(_f[0])
        parent_dim = int(_f[1])
        parent_tag = int(_f[2])
        num_partitions = int(_f[3])
        partitionTag =[int(l) for l in _f[4:4+num_partitions]]
        _ = 4+num_partitions+6 # to skip min and max
        num_physical_tags = int(_f[_])
        _ += 1
        physical_tags = dict_physicalTag(self,_f[_:_+num_physical_tags])
        _ = _+num_physical_tags
        num_bounding_curves = int(_f[_])
        _ += 1
        bounding_curves = [int(l) for l in _f[_:_+num_bounding_curves]]

def nodes(self, f):
    # Read the numbers of the section
    num_entity_blocks, numNodes, minNodeTag , maxNodeTag = f.readline().split()
    num_entity_blocks = int(num_entity_blocks)
    numNodes = int(numNodes)
    self.vertices =[None]*(numNodes+1)
    _node = 0
    for _entityBloc_b in range(num_entity_blocks):
        num_nodes_in_entity = int(f.readline().split()[3])
        list_nodes = [0]*num_nodes_in_entity
        for _ in range(num_nodes_in_entity):
            list_nodes[_] = int(f.readline())
        for _ in range(num_nodes_in_entity):
            coord = f.readline().split()
            coord = [float(__) for __ in coord]
            self.vertices[list_nodes[_]] = Vertex(coord, list_nodes[_])

def elements(self, f, p):
    num_entity_blocks, num_elements, min_element_tag, max_element_tag = readl_int(f)
    self.elements =[None]*(max_element_tag+1)
    for __ in range(int(num_entity_blocks)):
        entity_dim, entity_tag, element_type, num_elements_in_block = readl_int(f)
        if element_type not in self.reference_elements.keys():
            self.reference_elements[element_type] = reference_element(element_type, p.order)
        for _i in range(num_elements_in_block):
            element_tag, *node_tag = readl_int(f)
            if element_type == 1:
                if entity_dim != 1:
                    raise NameError("in import_msh_file, entity_dim!1 for element_type = 1")
                vertices = [self.vertices[n] for n in node_tag]
                self.elements[element_tag] = Element(element_type, element_tag, vertices, self.reference_elements[element_type])
            elif element_type == 2:
                if entity_dim != 2:
                    raise NameError("in import_msh_file, entity_dim!2 for element_type = 2")
                vertices = [self.vertices[n] for n in node_tag]
                self.elements[element_tag]=Element(element_type, element_tag, vertices, self.reference_elements[element_type])
            if isinstance(self.entities[self.entity_tag[entity_tag]], FemEntity) :
                self.entities[self.entity_tag[entity_tag]].elements.append(self.elements[element_tag])

def periodic(self, f):
    self.vertices_left = []
    self.vertices_right = []
    numPeriodicLinks = int(f.readline())
    for _link in range(numPeriodicLinks):
        entity_dim, entity_tag, entityTagMaster = readl_int(f)
        if entity_dim == 0:
            f.readline(); f.readline(); f.readline()
        if entity_dim == 1:
            f.readline()
            _ = int(f.readline())
            for i in range(_):
                r, l =readl_int(f)
                self.vertices_right.append(r)
                self.vertices_left.append(l)
            # check that all the nodes are on a vertical line
            x_left = self.vertices[self.vertices_left[0]].coord[0]
            coord_left =[self.vertices[_v].coord[0] for _v in self.vertices_left]
            if not np.allclose(x_left, coord_left):
                raise NameError("Error in import_msh_file: left boundary is not vertical")
            x_right = self.vertices[self.vertices_right[0]].coord[0]
            coord_right = [self.vertices[_v].coord[0] for _v in self.vertices_right]
            if not np.allclose(x_right, coord_right):
                raise NameError("Error in import_msh_file: right boundary is not vertical")
            period = abs(x_right-x_left)
            # period is defined on both the model and entities
            self.period = period
            for _ent in self.entities:
                if isinstance(_ent, (IncidentPwFem, TransmissionPwFem)):
                    _ent.period = period


def reference_element(typ, order):
    if typ == 2:
        out = Kt(order)
    elif typ == 1:
        out = Ka(order)
    return out

def readl_int(fid):
    return [int(_) for _ in fid.readline().split()]

def readl_float(fid):
    return [float(_) for _ in fid.readline().split()]
