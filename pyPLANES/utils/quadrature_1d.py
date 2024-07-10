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
# 	Mathieu Gaborit <gaborit@univ-lemans.fr>

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
from scipy import integrate
# from .gauss_kronrod import gauss_kronrod, import_gauss_kronrod, 
from .quadrature_schemes import import_scheme


class ReferenceScheme():
    def __init__(self, typ, order=3):
        
       
        self.xi_c, self.xi_powers_c, self.w_c, self.xi_r, self.xi_powers_r, self.w_r, self.Mat_c, self.Mat_r, self.mat_c, self.mat_r = import_scheme(typ, order)
        self.indices_c = np.array([i for i in range(0, 2*order+1, 2)])


        # self.xi_c, self.w_c = np.array(sc["xi_c"]), np.array(sc["w_c"])
        # self.xi_r, self.w_r = np.array(sc["xi_r"]), np.array(sc["w_r"])
        self.n_c, self.n_r = len(self.xi_c), len(self.xi_r)

    def __str__(self):
        s = f"Reference scheme\n"
        s += f"xi_c: {self.xi_c}\n"
        s += f"xi_r: {self.xi_r}\n"
        return s

class Integral(): 
    def __init__(self, f, a=0, b=pi/2, typ="CC"):

        self.f = f
        self.a = a 
        self.b = b
        self.typ = typ
        self.reference_scheme = ReferenceScheme(typ, 2)
        self.interval_list = [Interval(self.a, self.b, self.typ, self.reference_scheme)]
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
            quad_int.update(func, it)
            I_true = self.f(quad_int.b)-self.f(quad_int.a)
            print(f"e_c, e_r={np.abs(quad_int.I_c-I_true), np.abs(quad_int.I_r-I_true)}")
            self.I_c += quad_int.I_c
            self.I_r += quad_int.I_r
            self.error_list.append(quad_int.error)
        self.error = np.sum(self.error_list)
        self.error_list /= self.error

    def refine(self, it):
        interval_list = []
        for i_interval, quad_int in enumerate(self.interval_list):
            interval_list.extend(quad_int.split())
        self.interval_list = interval_list
            
class Interval():
    def __init__(self, a, b, typ, reference_scheme, previous_values=None, **kwargs):
        # print(f"Interval [{a}, {b}]")
        self.a = a
        self.b = b
        self.typ = typ
        self.reference_scheme = reference_scheme
        self.n_c, self.n_r = reference_scheme.n_c, reference_scheme.n_r
        self.indices_c = reference_scheme.indices_c
        self.xi_power_c = reference_scheme.xi_powers_c
        self.xi_power_r = reference_scheme.xi_powers_r
        self.mat_c = reference_scheme.mat_c
        self.mat_r = reference_scheme.mat_r
        self.xi_c = reference_scheme.xi_c
        self.xi_r = reference_scheme.xi_r

        # Determination of xi_a, xi_b, x__A and x_B
        if previous_values is None:
            xi_a, xi_b = -1, 1
            self.A, self.B = self.a, self.b
        else:
            self.previous_values = previous_values
            # Expressions page 48
            if self.previous_values["type"] == "left": # refinement of the first interval
                xi_0, xi_1 = self.xi_c[0], self.xi_r[-1]
            elif self.previous_values["type"] == "middle":
                xi_0, xi_1 = self.xi_r[0], self.xi_r[-1]
            elif self.previous_values["type"] == "right": # refinement of the last interval
                xi_0, xi_1 = self.xi_r[0], self.xi_c[-1]
            else:
                raise NameError("type should be left, right or middle") 
            alpha = (xi_1-xi_0)/(self.previous_values["x_r"][-1]-self.previous_values["x_r"][0])    
            beta = xi_0-alpha*self.previous_values["x_r"][0]
            
            xi_a, xi_b = alpha*self.a+beta, alpha*self.b+beta
            self.A, self.B = (-1-beta)/alpha, (1-beta)/alpha

        # self.a, self.b = None, None
        self.xia_power_c = np.array([(xi_a)**i for i in range(0, self.n_c+1)])
        self.xia_power_r = np.array([(xi_a)**i for i in range(0, self.n_r+1)])
        self.xib_power_c = np.array([(xi_b)**i for i in range(0, self.n_c+1)])
        self.xib_power_r = np.array([(xi_b)**i for i in range(0, self.n_r+1)])

        self.x_c = self.A+(self.B-self.A)*(reference_scheme.xi_c+1)/2.
        self.x_r = self.A+(self.B-self.A)*(reference_scheme.xi_r+1)/2.

        self.Mat_c = ((self.B-self.A)/2)*reference_scheme.Mat_c
        self.Mat_r = ((self.B-self.A)/2)*reference_scheme.Mat_r




    def __str__(self):
        s = f"interval [{self.a} ; {self.b}]\n"
        if self.previous_values is not None:
            s += f"previous values: {self.previous_values}\n"
        return s

    def update(self, func, it):
        self.f_r = np.array([func(x) for x in self.x_r])
        self.f_c = self.f_r[self.indices_c]

        # Coefficients of the antideratives of the interpolating polynomial

        self.coeff_c = (self.Mat_c@self.f_c)[::-1]
        self.coeff_r = (self.Mat_r@self.f_r)[::-1]
        # Computation of the integrals

        self.I_c = self.coeff_c@(self.xib_power_c-self.xia_power_c)
        self.I_r = self.coeff_r@(self.xib_power_r-self.xia_power_r)
        # self.error = np.abs((self.I_c-self.I_r)/self.I_r)
        
        # Computation of the errors
        if self.typ == "GK": 
            x_c = np.hstack([self.a, self.x_c, self.b])
        elif self.typ == "CC":
            x_c = self.x_c
        else:
            raise NameError("Unknown quadrature scheme")
        # self.error_list_0 = np.abs(np.array([poly_error(x_c[i+1])-poly_error(x_c[i]) for i in range(self.n_c+1)]))
        # self.error_list_0 /= np.sum(self.error_list_0)
        print(x_c)
        self.error_list = np.abs(np.array([ (self.coeff_c@self.xi_power_c[i+1]-self.coeff_r@self.xi_power_r[i+1])-(self.coeff_c@self.xi_power_c[i]-self.coeff_r@self.xi_power_r[i])for i in range(self.n_c+1)]))
        self.error_list /= np.sum(self.error_list)

        # self.x_plot =np.linspace(self.a, self.b, 500)
        # self.xi_plot = 2*(self.x_plot-self.x_A)/(self.Delta)-1
        # self.v_plot = [func(xx) for xx in self.x_plot]

        # # Coefficients of the interpolating polynomial
        # c_c = self.mat_c@f_c
        # c_r = self.mat_r@self.f_r

        # plt.figure(it)
        # plt.plot(self.x_r, self.f_r, "b.")
        # plt.plot(self.x_c, [c_c@xi[1:] for xi in self.xi_power_c[1:-1]], "+")
        # plt.plot(self.x_c, [c_r@xi[1:] for xi in self.xi_power_r[1:-1]], "+")
        # # plt.plot(self.x_plot, poly_r(self.x_plot), "r", label="poly_f")
        # # plt.plot(self.x_r, self.f_r, "r.")
        # plt.plot(self.x_plot, self.v_plot, "m", label="Tau")
        # plt.plot([self.a, self.a], [0, 3], color="k")
        # plt.plot([self.b, self.b], [0, 3], color="k")
        # def poly_r(xi):
        #     return np.sum([c_r[::-1][i]*xi**i for i in range(self.n_r)])
        # p_plot = [poly_r(xi) for xi in self.xi_plot]
        # plt.plot(self.x_plot, p_plot, "g", label="Tau")

        
        
        
        # plt.plot(self.x_plot, poly_diff(self.x_plot), "g", label="poly_error")
        # plt.plot(self.x_c, poly_diff(self.x_c), "g.")
        # plt.legend()
        # plt.figure(10+it)
        # plt.bar(range(self.n_c+1), self.error_list)
        # plt.plot(range(self.n_c+1), self.error_list,'r.-')
        # plt.show()

    def split(self):
        # detemine the number of nodes to add in each interval
        nb_sub_intervals = len(self.x_c)+1
        list_subintervals = [None]*nb_sub_intervals
        # index_max = np.argmax(self.error_list)

        # First interval
        previous_values = {"x_r" : self.x_r[:2], "f": self.f_r[:2], "xi" : self.xi_r[:2], "type" : "left"}
        list_subintervals[0] = Interval(self.a, self.x_c[0], self.typ, self.reference_scheme, previous_values)
        # Intermediate intervals
        for i in range(nb_sub_intervals-2):
            previous_values = {"x_r" : self.x_r[1+2*i:1+2*i+3], "f": self.f_r[1+2*i:1+2*i+2], "xi" : self.xi_r[1+2*i:1+2*i+3], "type" : "middle"}
            list_subintervals[i+1] = Interval(self.x_c[i], self.x_c[i+1], self.typ, self.reference_scheme, previous_values)
        # Last interval
        previous_values = {"x_r" :  self.x_r[-2:], "f": self.f_r[-2:], "xi": self.xi_r[-2:], "type" : "right"}
        list_subintervals[nb_sub_intervals-1] = Interval(self.x_c[-1], self.b, self.typ, self.reference_scheme, previous_values)
        
        
        # for interval in list_subintervals:
        #     print(interval.a, interval.b)
        #     print(interval.previous_values["x_r"])
        # sfdsdffdsfds
        return list_subintervals



    def plot(self):
        x_plot =np.linspace(self.A, self.B, 500)
        xi_plot =np.linspace(-1, 1, 500)
        v_plot = [np.exp(xx) for xx in x_plot]
        # plt.plot(x_plot, v_plot, "b")
        c_c = (self.mat_c@self.f_c)[::-1]
        c_r = (self.mat_r@self.f_r)[::-1]
        erreur_c_plot = (np.array([np.sum([c_c[i]*xi**i for i in range(self.n_c)]) for xi in xi_plot])-v_plot)
        erreur_r_plot = (np.array([np.sum([c_r[i]*xi**i for i in range(self.n_r)]) for xi in xi_plot])-v_plot)
        plt.plot(x_plot,erreur_c_plot,"g")
        plt.plot(x_plot,erreur_r_plot,"m")
        
        
        
        # plt.plot(self.x_c, [self.mat_c@self.f_r[self.indices_c]@xi[1:] for xi in self.xi_power_c[1:-1]], "+")
        # plt.plot(self.x_c, [self.mat_r@self.f_r@xi[1:] for xi in self.xi_power_r[1:-1]], "+")
        # plt.plot([self.a, self.a], [0, 3], color="k")
        # plt.plot([self.b, self.b], [0, 3], color="k")



    # def refine(self):
    #     # Determine on which interval the error is maximal
    #     index_max = np.argmax(self.error_list)
    #     print(index_max)
    #     if self.error_list[index_max]>0.5:
    #         # The error is on a single element
    #         if index_max in [0, self.n_c-2]: # The error is mainly on the first or last interval
    #             if index_max == 0:
    #                 x_cut = self.x_c[1]
    #             else:
    #                 x_cut = self.x_c[index_max]
    #             return [(self.a, x_cut), (x_cut,self.b)]
    #         else:
    #             x_minus, x_plus = self.x_c[index_max:index_max+2]
    #             return [(self.a, x_minus), (x_minus, x_plus), (x_plus, self.b)]
    #     else: # The error is globally well balanced between elements
    #         interval_list = [(self.a, self.x_c[0])]
    #         interval_list.extend([(self.x_c[i], self.x_c[i+1]) for i in range(self.n_c-1)])
    #         interval_list += [(self.x_c[-1], self.b)]
    #         return interval_list
