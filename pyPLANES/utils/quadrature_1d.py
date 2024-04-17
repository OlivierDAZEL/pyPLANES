#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# quadrature_1d.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2024
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
import matplotlib.pyplot as plt
from numpy import pi
from scipy.interpolate import lagrange
from pyPLANES.utils.gauss_kronrod import gauss_kronrod


# class ReferenceScheme():
#     def __init__(self, xi, w):
#         self.xi = xi
#         self.w = w

class AdaptativeReferenceScheme():
    def __init__(self, order=3):
        
        sc = gauss_kronrod[f"{order}"]
        self.xi_c, self.w_c = np.array(sc["xi_c"]), np.array(sc["w_c"])
        self.xi_r, self.w_r = np.array(sc["xi_r"]), np.array(sc["w_r"])
        self.n_c, self.n_r = len(self.xi_c), len(self.xi_r)
        self.indices_c = np.array(sc["indices_c"])

class Subdivision(): 
    def __init__(self, a=0, b=pi/2):
        
        self.interval_list = [QuadratureInterval(a, b, AdaptativeReferenceScheme(7))]
        self.compute_list = [True]
        self.error_list = []
        self.I_c, self.I_r, self.error = None, None, None

        
    def __str__(self):
        s = ""
        for i, sub in enumerate(self.interval_list):
            s+= f"Subdivision #{i}\n"
            s += sub.__str__()
        return s 

    def update(self, func, it):
        # Initialisation
        self.I_c, self.I_r, self.error, self.error_list = 0. , 0., 0, []
        for i_interval, quad_int in enumerate(self.interval_list):
            if self.compute_list[i_interval]:
                quad_int.update(func, it)
            self.I_c += quad_int.I_c
            self.I_r += quad_int.I_r
            self.error_list.append(quad_int.error)
        self.error = np.sum(self.error_list)
        self.error_list /= self.error

    def refine(self, it):
        new_interval_list = []
        new_compute_list = []
        index_max = np.argmax(self.error_list)
        refined_reference_scheme = AdaptativeReferenceScheme(3)
            
        if self.error_list[index_max]>0.5:
            if index_max == 0:
                # The first quadrature interval 
                quad_int = self.interval_list[0]
                for _int in quad_int.refine():
                    new_interval_list.append(QuadratureInterval(_int[0], _int[1], refined_reference_scheme))
                    new_compute_list.append(True)
                new_interval_list += self.interval_list[1:]
                new_compute_list += self.compute_list[1:]
            elif index_max == len(self.error_list)-1:
                new_interval_list += self.interval_list[:-1]
                new_compute_list+= self.compute_list[:-1]
                quad_int = self.interval_list[-1]
                for _int in quad_int.refine():
                    new_interval_list.append(QuadratureInterval(_int[0], _int[1], refined_reference_scheme))
                    new_compute_list.append(True)
            else:
                new_interval_list += self.interval_list[:index_max]
                new_compute_list +=self.compute_list[:index_max]
                quad_int = self.interval_list[index_max]
                for _int in quad_int.refine():
                    new_interval_list.append(QuadratureInterval(_int[0], _int[1], refined_reference_scheme))
                    new_compute_list.append(True)
                new_interval_list +=self.interval_list[index_max+1:]
                new_compute_list +=self.compute_list[index_max+1:]
        else:
            for quad_int in self.interval_list:
                list_intervals = quad_int.refine()
                for _int in list_intervals:
                    new_interval_list.append(QuadratureInterval(_int[0], _int[1], refined_reference_scheme))
                    new_compute_list.append(True)
        self.interval_list = new_interval_list
        self.compute_list = new_compute_list

class QuadratureInterval():
    def __init__(self, a, b, reference_scheme=AdaptativeReferenceScheme(), poly=np.poly1d([0]), **kwargs):
        
        self.a = a
        self.b = b
        self.poly = poly
        # integrations schemes
        if isinstance(reference_scheme, AdaptativeReferenceScheme):
            self.n_c, self.n_r = reference_scheme.n_c, reference_scheme.n_r
            self.indices_c = reference_scheme.indices_c
            
            self.x_r = self.a+(self.b-self.a)*(reference_scheme.xi_r+1)/2
            self.x_c = self.a+(self.b-self.a)*(reference_scheme.xi_c+1)/2
            self.w_c =  (self.b-self.a)*reference_scheme.w_c/2
            self.w_r =  (self.b-self.a)*reference_scheme.w_r/2

    def __str__(self):
        s = f"interval [{self.a} ; {self.b}]\n" 
        return s

    def calcul_int(self, x, f):
        Poly = np.polyint(lagrange(x, f))
        return Poly(self.b)-Poly(self.a)

    def update(self, func, it):
        f_r = np.array([func(x) for x in self.x_r])
        f_c = f_r[self.indices_c]

        self.I_c = np.sum(self.w_c*f_c)
        self.I_r = np.sum(self.w_r*f_r)
        self.error = np.abs((self.I_c-self.I_r)/self.I_r)


        poly_c = lagrange(self.x_c, f_c)
        poly_r = lagrange(self.x_r, f_r)
        
        self.poly_c = lagrange(self.x_c, f_c)
        self.poly_r = lagrange(self.x_r, f_r)
        self.f_c = f_c 
        self.f_r = f_r
        
        
        poly_diff = (poly_r-poly_c)
        poly_error = np.polyint(poly_diff)
        

        # self.x_plot =np.linspace(self.a, self.b, 500)
        # self.v_plot = [func(xx) for xx in self.x_plot]
        # plt.figure(it)
        # plt.plot(self.x_plot, poly_c(self.x_plot), "b", label="poly_c")
        # plt.plot(self.x_c, f_c, "b.")
        # plt.plot(self.x_plot, poly_r(self.x_plot), "r", label="poly_f")
        # plt.plot(self.x_r, f_r, "r.")
        # plt.plot(self.x_plot, self.v_plot, "m", label="Tau")
        # plt.plot(self.x_plot, poly_diff(self.x_plot), "g", label="poly_error")
        # plt.plot(self.x_c, poly_diff(self.x_c), "g.")
        # plt.legend()

        self.error_list = np.abs(np.array([poly_error(self.x_c[i+1])-poly_error(self.x_c[i]) for i in range(self.n_c-1)]))
        self.error_list /= np.sum(self.error_list)

        # plt.figure(10+it)
        # plt.bar(range(self.n_c-1), self.error_list)
        # plt.show()

    def refine(self):
        # Determine on which interval the error is maximal
        index_max = np.argmax(self.error_list)
        if self.error_list[index_max]>0.5:
            # The error is on a single element
            if index_max in [0, self.n_c-2]: # The error is mainly on the first or last interval
                if index_max == 0:
                    x_cut = self.x_c[1]
                else:
                    x_cut = self.x_c[index_max]
                return [(self.a, x_cut), (x_cut,self.b)]
            else:
                x_minus, x_plus = self.x_c[index_max:index_max+2]
                return [(self.a, x_minus), (x_minus, x_plus), (x_plus, self.b)]
        else: # The error is globally well balanced between elements
            interval_list = [(self.a, self.x_c[0])]
            interval_list.extend([(self.x_c[i], self.x_c[i+1]) for i in range(self.n_c-1)])
            interval_list += [(self.x_c[-1], self.b)]
            return interval_list
