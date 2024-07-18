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
import sys
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
        # Nodes
        nodeTags, nodeCoords, nodeParams =gmsh.model.mesh.getNodes()
        nb_nodes = len(nodeTags)
        self.vertices =[None]*(nb_nodes+1)
        nodeCoords = nodeCoords.reshape((nb_nodes,3))
        for i in range(nb_nodes):
            self.vertices[nodeTags[i]] = FemVertex(nodeCoords[i,:], nodeTags[i])
        # Elements
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
        nb_elements = int(max([max(a) for a in elemTags]))
        self.elements =[None]*(nb_elements+1)  
        for i_typ, typ in enumerate(elemTypes):
            tags = elemTags[i_typ]
            if typ ==1: # 2 nodes segments
                nodes = np.array(elemNodeTags[i_typ]).reshape((len(tags),2))
                for i, tag in enumerate(tags):
                    self.elements[tag] = FemElement(typ, tag, [self.vertices[j] for j in nodes[i,:]])
            if typ ==2: # TR3
                nodes = np.array(elemNodeTags[i_typ]).reshape((len(tags),3))
                for i, tag in enumerate(tags):
                    self.elements[tag] = FemElement(typ, tag, [self.vertices[j] for j in nodes[i,:]])
        # Entities (import them all first)
        entities = gmsh.model.getEntities()
        self.entities = dict()
        for ie, e in enumerate(entities):
            dim, tag = e

            physicalTags = [gmsh.model.getPhysicalName(dim, p) for p in gmsh.model.getPhysicalGroupsForEntity(dim, tag)]
            # conversion of physicalTags to a dict
            list_physicalTags =  [_.split("=") for _ in physicalTags]
            keys = [_[0] for _ in list_physicalTags]
            values = [_[1] for _ in list_physicalTags]
            physicalTags = dict(zip(keys, values))
            # end of conversion of physicalTags to a dict
            # Determine the entity constructor
            if dim ==0:
                entity_constructor = GmshEntity
            elif dim == 1:
                if len(physicalTags) !=0:
                    if (physicalTags["condition"]=="bottom") or (physicalTags["condition"]=="top"):
                        entity_constructor= PwFem
                    elif physicalTags["condition"]=="Periodicity":
                        entity_constructor= PeriodicityFem
                    elif physicalTags["condition"]=="Imposed displacement":
                        entity_constructor= ImposedDisplacementFem
                    else:
                        entity_constructor= GmshEntity
            elif dim == 2:
                mat = load_material(physicalTags["mat"])
                if mat.MEDIUM_TYPE in ["eqf", "fluid"]:
                    entity_constructor = FluidFem
                elif mat.MEDIUM_TYPE in ["elastic"]:
                    entity_constructor = ElasticFem
                elif mat.MEDIUM_TYPE in ["pem"]:
                    entity_constructor = PemFem
                else:
                    raise NameError("invalid material")
            
            _ent = entity_constructor(dim=dim, tag=tag, physicalTags=physicalTags, entities=self.entities, updown=gmsh.model.getAdjacencies(dim, tag))
            
            # Affect elements to fem entities
            if isinstance(_ent, FemEntity):
                elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)
                for e in elemTags[0]:
                    _ent.elements.append(self.elements[e])

            
          # Add entity to lists
            self.entities[(dim,tag)] = _ent
            if isinstance(_ent, FemEntity):
                if isinstance(_ent, PwFem):
                    self.pwfem_entities.append(_ent)
                else:
                    self.fem_entities.append(_ent)
  
  
  
        # periodicity
        for e in self.fem_entities:
            if isinstance(e, PeriodicityFem):
                masterTag, nodeTags, nodeMasterTags, tfo = gmsh.model.mesh.getPeriodicNodes(e.dim, e.tag, includeHighOrderNodes=False)
                if len(nodeTags) != 0:
                    self.period = tfo[3]
                    if self.period > 0:
                        self.vertices_left = nodeMasterTags
                        self.vertices_right = nodeTags
                    else:
                        self.vertices_left = nodeTags
                        self.vertices_right = nodeMasterTags
        if hasattr(self, 'period'):
            self.period = np.abs(self.period)
            for e in self.entities.values():
                if isinstance(e, PwFem):
                    e.period = self.period




        # updates entities 
        # replace up and down variables by entities instead of GMSH tags
        for e in self.entities.values():
            e.up = [self.entities[(e.dim+1, tag)] for tag in e.up] 
            e.down = [self.entities[(e.dim-1, tag)] for tag in e.down]
            if hasattr(e, "mat"):
                e.mat = load_material(e.mat)


        

        gmsh.clear()
        gmsh.finalize()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
 