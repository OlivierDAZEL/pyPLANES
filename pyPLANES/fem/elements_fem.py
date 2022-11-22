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
import numpy.linalg as LA
from pyPLANES.generic.elements_generic import GenericVertex, GenericEdge, GenericElement
from pyPLANES.fem.lagrange_polynomials import lagrange_on_Kt, lagrange_on_Ka


class FemVertex(GenericVertex):
    '''Vertex Finite-Element'''
    def __init__(self, coord, tag):
        GenericVertex.__init__(self, coord, tag)
        self.dofs = [0] * 4
        self.sol = np.zeros(4,dtype=complex)
    def __str__(self):
        out = GenericVertex.__str__(self)
        out += "dofs=" + format(self.dofs)+"\n"
        return out

class FemEdge(GenericEdge):
    ''' TODO '''
    def __init__(self, tag, vertices, element, order):
        GenericEdge.__init__(self, tag, vertices, element)
        self.tag = tag
        self.vertices = vertices
        self.elements = [element]
        self.order = [order, order, order, order]
        self.dofs = [[0]*(order-1)]*4
        self.sol = [np.zeros((order-1))]*4

    def __str__(self):
        out = GenericEdge.__str__(self)
        out += "dofs=" + format(self.dofs)+"\n"
        return out

class FemFace:
    """ Finite-Element Face

    Parameters
    ----------
    tag : int
        GMSH tag
    vertices : list
        vertices linked to the element
    element : [type]
        [description]
    order : [type]
        [description]
    """

    def __init__(self, tag, vertices, element, order):
        self.tag = tag
        self.vertices = vertices
        self.elements = [element]
        self.order = [order, order, order, order]
        _ = [0]*int((order-1)*(order-2)/2)
        self.dofs = [_, _, _, _]
        self.sol = [np.zeros(int((order-1)*(order-2)/2))]*4
    def __str__(self):
        out = "Face #{}\n".format(self.tag)
        out  += "Vertices=[{},{},{}], ".format(self.vertices[0].tag, self.vertices[1].tag, self.vertices[2].tag)
        related_elements = [_el.tag for _el in self.elements]
        out  += "related elements={}\n".format(related_elements)
        out += "dofs=" + format(self.dofs)+"\n"
        return out

class FemBubble:
    ''' Class Bubble '''
    """
    [summary]
    """
    def __init__(self,nodes,element,order,geo):
        """
        [summary]

        Parameters
        ----------
        nodes : [type]
            [description]
        element : [type]
            [description]
        order : [type]
            [description]
        geo : [type]
            [description]
        """
        self.geo = geo
        self.nodes = nodes
        if hasattr(self, 'elements'):
            self.elements.append(elements)
        else:
            self.elements=[element]
        self.order = [order, order, order, order]
        self.dofs = [[],[],[],[]]
        self.sol = [np.zeros(int((order-1)*(order-2)*(order-3)/2))]*4

    def __str__(self):
        out = "geo/Nodes/Elements/order/dofs = " + str(self.geo)+"/" + str(self.nodes)+"/"+ format(self.elements)
        out += "/".format(self.order)
        out += format(self.dofs)+"\n"
        return out

class FemElement(GenericElement):
    ''' Element of pyPLANES

    Parameters:
    -----------
    typ : int
        GMSH type of the element

    coorde : numpy array
        Array of nodes coordinates (dim = 3x nb vertices )

    Ref_Elem : Reference Element


    Attributes :
    ------------------------

    edges : List of edge instances associated to the element

    faces : List of face instances associated to the element (optional)

    bubbles : List of bubble instances associated to the element (optional)

    '''
    
    def __init__(self, typ, tag, vertices):
        GenericElement.__init__(self, typ, tag, vertices)
        self.reference_element = None
        self.dofs = [[], [], [], []]
        # Rules for the dofs indices vector
        if typ in [1,8]: # Line Elements
            self.edges = []
            self.edges_orientation = []
            self.nb_edges = 0
        elif typ in [2, 9]: # triangles
            self.edges, self.faces = [], []
            self.edges_orientation, self.faces_orientation = [], []
        else: 
            raise NameError("typ {} is not a supported element".format(typ))

    def get_normal(self, xi_1, elem_2d):
        
        if self.typ == 1: #Line2
            JJ = lagrange_on_Ka(1, xi_1)[1]
            return np.abs(LA.norm(self.coord@JJ))
        elif self.typ == 8: # Line3
            # print("xi_1={}".format(xi_1))
            n_ = self.coord@lagrange_on_Ka(2, xi_1)[1]

            n_ = np.array([n_[1], -n_[0]])
            n_ = n_/LA.norm(n_)
            vec_2dto1d = self.get_center()-elem_2d.get_center()
            if np.dot(n_.reshape((2)), vec_2dto1d)<0:
                n_ *= -1
            return n_
        else: 
            raise NameError("Unknown type of element in fem/elements_fem/FemElement/get_normal")

    def get_jacobian_matrix(self, xi_1=0, xi_2=0):
        
        if self.typ == 1: #Line2
            JJ = lagrange_on_Ka(1, xi_1)[1]
            return np.abs(LA.norm(self.coord@JJ))
        elif self.typ == 8: # Line3
            # print("xi_1={}".format(xi_1))
            JJ = lagrange_on_Ka(2, xi_1)[1]
            return np.abs(LA.norm(self.coord@JJ))
        elif self.typ == 2: # TR3
            JJ = lagrange_on_Kt(1, xi_1, xi_2)[1]
            return JJ.dot(self.coord.T)
        elif self.typ == 9: # TR6
            JJ = lagrange_on_Kt(2, xi_1, xi_2)[1]
            return JJ.dot(self.coord.T)
        else: 
            raise NameError("Unknown type of element in fem/elements_fem/FemElement/get_jacobian_matrix")
        


    def __str__(self):
        out = "Element #{} / typ={} / reference element ={}\n".format(self.tag, self.typ, self.reference_element)
        if self.typ == 1: 
            out += "Vertices = [{},{}]\n".format(self.vertices[0].tag, self.vertices[1].tag)
            out += "edge={} /orientation ={}\n".format(self.edges[0].tag, self.edges_orientation)
        elif self.typ == 2:
            out += "Vertices = [{},{},{}]\n".format(self.vertices[0].tag, self.vertices[1].tag, self.vertices[2].tag)
            out += "edges =[{},{},{}]\n".format(self.edges[0].tag, self.edges[1].tag, self.edges[2].tag)
            out += "edge orientation ={}\n".format(self.edges_orientation)
        # out += "dofs={}".format(self.dofs)
        return out

    def display_sol(self, field):
        """
        Returns the values of the coordinates 

        Parameters
        ----------
        field : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        order = self.reference_element.order
        if not isinstance(field, list):
            field = [field]
        f_elem = np.zeros((self.reference_element.nb_SF, len(field)), dtype=complex)
        coorde = self.coord
        if self.typ == 2:
            for i_f, _field in enumerate(field):
                f_elem[0, i_f] = self.vertices[0].sol[_field]
                f_elem[1, i_f] = self.vertices[1].sol[_field]
                f_elem[2, i_f] = self.vertices[2].sol[_field]
                for ie, _edge in enumerate(self.edges):
                    f_elem[3+ie*(order-1):3+(ie+1)*(order-1), i_f] = _edge.sol[_field]*(self.edges_orientation[ie]**np.arange((order-1)))
                f_elem[3+3*(order-1):, i_f] = self.faces[0].sol[_field]

            x = np.dot(coorde[0, :].T, self.reference_element.phi_plot[:3, :])
            y = np.dot(coorde[1, :].T, self.reference_element.phi_plot[:3, :])
            f = np.dot(self.reference_element.phi_plot.T, f_elem)
            return x, y, f

    def elementary_matrices(self):
        pass




            # theta = np.pi/8
            # self.coord=np.array([[1,np.cos(theta),np.cos(theta/2)],[0,np.sin(theta),np.sin(theta/2)]])
            


            # linear =LA.norm(self.coord[:, 1]-self.coord[:, 0])/2
            # # print(self.coord)
            # print("ref")
            # ref = theta/2

            # print(ref)
            # print("error old= {}".format(np.abs(LA.norm(self.coord[:, 1]-self.coord[:, 0])/2.-ref)))
            # print("error new= {}".format(-ref)))
            

            # xi =np.linspace(-1, 1, 100)
            # N = lagrange_on_Ka(2, xi)[0]





            # X = self.coord@N
            # dX = self.coord@lagrange_on_Ka(2, xi)[1]
            # ds = np.sqrt(dX[0]**2+dX[1]**2)

            # import matplotlib.pyplot as plt

            # plt.figure()
            # # plt.plot(xi,X[0],'b',label="x")
            # # plt.plot(xi,X[1],'r',label="y")
            # plt.plot(xi,dX[0],'k',label="dx")
            # plt.plot(xi,dX[1],'m',label="dy")
            # plt.plot(xi,ds,'r',label="ds")
            # # plt.plot(xi, np.cos((xi+1)*theta/2),'b.')
            # # plt.plot(xi, np.sin((xi+1)*theta/2),'r.')

            # plt.plot(xi, -(theta/2)*np.sin((xi+1)*theta/2),'k.')
            # plt.plot(xi, (theta/2)*np.cos((xi+1)*theta/2),'m.')
            # plt.plot(xi, (theta/2)*xi**0,'r.')
            # plt.plot(xi, linear*xi**0,'b.',label="linear")

            # plt.title("{}".format(theta/2))
            # plt.legend()

            # plt.figure()
            # plt.plot(X[0],X[1],'r.')
            # plt.plot(np.cos(np.linspace(0, theta)),np.sin(np.linspace(0, theta)),'b')
            # plt.plot(self.coord[0,:],self.coord[1,:],"b.")
            # plt.axis("equal")
            # plt.show()



            # dssfdfdsfds