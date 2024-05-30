#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# import_msh_file.py
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

from mediapack import Air, Fluid
from pyPLANES.utils.io import load_material

from pyPLANES.fem.elements_fem import FemVertex, FemElement
from pyPLANES.dgm.dgm_elements import DgmElement


from pyPLANES.core.mesh import Mesh
from pyPLANES.generic.entities_generic import GmshEntity, FemEntity, DgmEntity
from pyPLANES.fem.fem_entities_surfacic import ImposedDisplacementFem, FluidStructureFem, RigidWallFem, InterfaceFem, PeriodicityFem, RobinAirFem, ImposedPwFem
from pyPLANES.fem.fem_entities_volumic import FluidFem, ElasticFem, PemFem
from pyPLANES.fem.fem_entities_pw import PwFem
from pyPLANES.dgm.dgm_entities_surfacic import ImposedDisplacementDgm
from pyPLANES.dgm.dgm_entities_volumic import FluidDgm
import gmsh


class GmshMesh(Mesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_msh_file(self):
    
        gmsh.initialize()
        gmsh.open(self.msh_file)
        
        # physical_name= gmsh.model.get_physical_name()
        # print(physical_name)
        
        entities = gmsh.model.getEntities()
        print(entities)
        jk
        
        
        if self.verbose:
            print(f"Reading {self.msh_file}")
        f = open(self.msh_file, "r")
        _f = "_"
        while _f:
            _f = f.readline().strip()
            if _f.startswith('$'):
                tag = _f[1:]
                if self.verbose:
                    print(f"Tag = {tag}")
                if tag == "MeshFormat":
                    self.MeshFormat = f.readline()
                    
                    
                    qdssqd
                    if self.verbose:
                        print(f"MeshFormat={self.MeshFormat}")
                if tag == "PhysicalNames":
                    self._physical_names(f)

                if tag == "Entities":
                    self._entities(f)
                # if tag == "PartitionedEntities":
                #     self._partition(f)
                if tag == "Nodes":
                    self._nodes(f)
                if tag == "Elements":
                    self._elements(f)
                if tag == "Periodic":
                    self._periodic(f)
                _ = f.readline()
                if _.strip() != "$End"+ tag:
                    raise NameError("Error in GMSH file importation at tag:" +tag)
        # for e in self.fem_entities:
        #     e.print_elements()

    def _dict_physical_tags(self, _list):
        ''' create a dict from gmsh file physical tags.
            Possible keys: model and materials'''
        d = [self.physical_names[int(_)] for _ in _list]
        d = [_.split("=") for _ in d]
        key_list = [_[0] for _ in d]
        value_list = [_[1] for _ in d]
        d = dict(zip(key_list, value_list))
        return d

    def _physical_names(self, f):
        ''' Importation of Physical names'''
        num_physical_names = int(f.readline())
        if self.verbose:
            print(f"{num_physical_names} Physical names")
        self.physical_names = dict()
        for __ in range(num_physical_names):
            _ = f.readline().split()
            tag = int(_[1])
            key = " ".join(_[2:])[1:-1]
            self.physical_names[tag] = key
        if self.verbose:
            print(f"{self.physical_names}")


    def _entities(self, f):
        ''' creation of the list of entities '''
        _p, num_curves, num_surfaces, num_volumes = self.readl_int(f)

        if self.verbose:
            print(f"{_p} points ")
        for __ in range(_p):
            _f = f.readline().split()
            # Read the tag of the point and then the x,y,z, coordinates
            tag = int(_f[0])
            x, y, z = float(_f[1]), float(_f[2]), float(_f[3])
            # Read the physical tags
            num_physical_tags = int(_f[4])
            physical_tags = self._dict_physical_tags(_f[8:8+num_physical_tags])
            _ent = GmshEntity(dim=0, tag=tag, physical_tags=physical_tags, x=x, y=y, z=z, entities=self.entities)
            self.entities.append(_ent)
        # print("curves")
        if self.verbose:
            print(f"{num_curves} curves ")
        for _icurve in range(num_curves):
            _f = f.readline().split()
            tag = int(_f[0])
            num_physical_tags = int(_f[7])
            physical_tags = self._dict_physical_tags(_f[8:8+num_physical_tags])
            _ = 8+num_physical_tags
            num_bounding_points = int(_f[_])
            bounding_points = [int(_l) for _l in _f[_+1:]] if num_bounding_points != 0 else []
            if "typ" in physical_tags.keys():
                if self.verbose:
                    print(f"Curve #{_icurve} / Tag# {tag}")
                if physical_tags["typ"] == "1D":
                    if physical_tags["condition"].lower() in ["top", "bottom"]:
                        if physical_tags["method"] == "FEM":
                            _ent = PwFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                            if physical_tags["condition"].lower() == "bottom":
                                _ent.ny = -1.
                            self.pwfem_entities.append(_ent)
                    elif physical_tags["condition"] == "Imposed displacement":
                        if physical_tags["method"] == "FEM":
                            _ent = ImposedDisplacementFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                            self.fem_entities.append(_ent)
                    elif physical_tags["condition"] == "Imposed PW":
                        if physical_tags["method"] == "FEM":
                            _ent = ImposedPwFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                            self.fem_entities.append(_ent)
                    elif physical_tags["condition"] == "Robin Air":
                        if physical_tags["method"] == "FEM":
                            _ent = RobinAirFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                            self.fem_entities.append(_ent)
                        elif physical_tags["method"] == "DGM":
                                _ent = ImposedDisplacementDgm(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                                self.dgm_entities.append(_ent)
                    elif physical_tags["condition"].lower() in ["fluid_structure", "ifs"]:
                        if physical_tags["method"] == "FEM":
                            _ent = FluidStructureFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                            self.fem_entities.append(_ent)
                    elif physical_tags["condition"].lower() in ["rigid wall", "rigid", "wall"]:
                        if physical_tags["method"] == "FEM":
                            _ent = RigidWallFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                            # print(_ent)
                            # print(_ent.tag)
                        elif physical_tags["method"] == "DGM":
                            _ent = RigidWallFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                            self.dgm_entities.append(_ent)
                    elif physical_tags["condition"] == "Periodicity":
                        if physical_tags["method"] == "FEM":
                            _ent = PeriodicityFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities,verbose=self.verbose)
                            self.fem_entities.append(_ent)
                    elif physical_tags["condition"].split("/")[0] == "Interface":
                        if physical_tags["method"] == "FEM":
                            _ent = InterfaceFem(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities, ml_name=physical_tags["condition"].split("/")[1], side=physical_tags["condition"].split("/")[2])
                            self.fem_entities.append(_ent)
                    else:
                        raise NameError("FEM1D entity without physical condition")

            else: # No numerical model
                _ent = GmshEntity(dim=1, tag=tag, physical_tags=physical_tags, bounding_points=bounding_points, entities=self.entities)
            self.entities.append(_ent)
        # print("surfaces")
        if self.verbose:
            print(f"{num_surfaces} surfaces ")
        for _surf in range(num_surfaces):
            _f = f.readline().split()
            tag = int(_f[0])
            num_physical_tags = int(_f[7])
            physical_tags = self._dict_physical_tags(_f[8:8+num_physical_tags])

            _ = 8+num_physical_tags
            num_bounding_curves = int(_f[_])
            bounding_curves = [int(_l) for _l in _f[_+1:]] if num_bounding_curves != 0 else []
            if "typ" in physical_tags.keys():
                if self.verbose:
                    print(f"Surface #{_surf} / Tag# {tag}")
                    print(physical_tags)
                if physical_tags["typ"] == "2D":
                    if "mat" in physical_tags.keys():
                        if physical_tags["mat"].split()[0] == "Air":
                            mat = Fluid(c=Air().c,rho=Air().rho)
                            if physical_tags["method"] == "FEM":
                                _ent = FluidFem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, mat=mat, entities=self.entities, condensation=self.condensation,verbose=self.verbose)
                                self.fem_entities.append(_ent)
                            elif physical_tags["method"] == "DGM":
                                _ent = FluidDgm(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, mat=mat, entities=self.entities,verbose=self.verbose)
                                self.dgm_entities.append(_ent)
                        else:
                            mat = load_material(physical_tags["mat"].split()[0])
                            if mat.MEDIUM_TYPE in ["eqf", "fluid"]:
                                if physical_tags["method"] == "FEM":
                                    _ent = FluidFem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, mat=mat, entities=self.entities, condensation=self.condensation,verbose=self.verbose)
                                    self.fem_entities.append(_ent)
                            elif mat.MEDIUM_TYPE == "elastic":
                                if physical_tags["method"] == "FEM":
                                    _ent = ElasticFem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, mat=mat, entities=self.entities, condensation=self.condensation,verbose=self.verbose)
                                    self.fem_entities.append(_ent)
                            elif mat.MEDIUM_TYPE == "pem":
                                if physical_tags["method"] == "FEM":
                                    _ent = PemFem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, mat=mat, entities=self.entities, condensation=self.condensation,verbose=self.verbose)
                                    self.fem_entities.append(_ent)
                            else:
                                raise NameError(" Provided material is neither eqf, elastic nor pem")

                # elif physical_tags["typ"] == "0D":
                #     _ent = FluidFem(dim=2, tag=tag, physical_tags=physical_tags, bounding_curves=bounding_curves, mat=Air, entities=self.entities)
                #     self.fem_entities.append(_ent)
            self.entities.append(_ent)

        for __ in range(num_volumes):
            pass

        _ = [_ent.tag for _ent in self.entities]
        self.entity_tag = [None]*(max(_)+1)
        for i, index in enumerate(_):
            self.entity_tag[index] = i

    def _partition(self, f):
        self.PartitionedEntities = []
        num_partitions = int(f.readline())
        num_ghostEntities = int(f.readline())
        num_points, num_curves, num_surfaces, num_volumes = self.readl_int(f)
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
            physical_tags = self._dict_physicalTag(_f[_:_+num_physical_tags])
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
            physical_tags = self._dict_physicalTag(_f[_:_+num_physical_tags])
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
            physical_tags = self._dict_physicalTag(_f[_:_+num_physical_tags])
            _ = _+num_physical_tags
            num_bounding_curves = int(_f[_])
            _ += 1
            bounding_curves = [int(l) for l in _f[_:_+num_bounding_curves]]

    def _nodes(self, f):
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
                self.vertices[list_nodes[_]] = FemVertex(coord, list_nodes[_])

    def _elements(self, f):
        num_entity_blocks, num_elements, min_element_tag, max_element_tag = self.readl_int(f)
        self.elements =[None]*(max_element_tag+1)
        for __ in range(int(num_entity_blocks)):
            entity_dim, entity_tag, element_type, num_elements_in_block = self.readl_int(f)
            for _i in range(num_elements_in_block):
                element_tag, *node_tag = self.readl_int(f)
                if element_type in [1, 8]:
                    if entity_dim != 1:
                        raise NameError("in load_msh_file, entity_dim!=1 for element_type = {}".format(element_type))
                    vertices = [self.vertices[n] for n in node_tag]
                elif element_type in [2, 9]:
                    if entity_dim != 2:
                        raise NameError("in import_msh_file, entity_dim!=2 for element_type =  {}".format(element_type))
                    vertices = [self.vertices[n] for n in node_tag]
                else:
                    raise NameError("{} is an incompatible type of element".format(element_type))
                if isinstance(self.entities[self.entity_tag[entity_tag]], FemEntity):
                    self.elements[element_tag] = FemElement(element_type, element_tag, vertices)
                    self.entities[self.entity_tag[entity_tag]].elements.append(self.elements[element_tag])
                elif isinstance(self.entities[self.entity_tag[entity_tag]], DgmEntity):
                    self.elements[element_tag] = DgmElement(element_type, element_tag, vertices)
                    self.entities[self.entity_tag[entity_tag]].elements.append(self.elements[element_tag])               

    def _periodic(self, f):
        self.vertices_left = []
        self.vertices_right = []
        numPeriodicLinks = int(f.readline())
        for _link in range(numPeriodicLinks):
            entity_dim, entity_tag, entityTagMaster = self.readl_int(f)
            if entity_dim == 0:
                f.readline(); f.readline(); f.readline()
            if entity_dim == 1:
                f.readline()
                _ = int(f.readline())
                for i in range(_):
                    r, l =self.readl_int(f)
                    self.vertices_right.append(r)
                    self.vertices_left.append(l)
                # check that all the nodes are on a vertical line
                x_left = self.vertices[self.vertices_left[0]].coord[0]
                coord_left = [self.vertices[_v].coord[0] for _v in self.vertices_left]
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
                    if isinstance(_ent, PwFem):
                        _ent.period = period

    def readl_int(self, fid):
        return [int(_) for _ in fid.readline().split()]
