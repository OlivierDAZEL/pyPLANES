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
from scipy import integrate
from .gauss_kronrod import gauss_kronrod, import_gauss_kronrod


class ReferenceScheme():
    def __init__(self, order=3):
        
        sc = gauss_kronrod[f"{order}"]
        self.xi_c, self.xi_powers_c, self.w_c, self.xi_r, self.xi_powers_r, self.w_r, self.Mat_c, self.Mat_r = import_gauss_kronrod(order)
        self.xi_c, self.w_c = np.array(sc["xi_c"]), np.array(sc["w_c"])
        self.xi_r, self.w_r = np.array(sc["xi_r"]), np.array(sc["w_r"])
        self.n_c, self.n_r = len(self.xi_c), len(self.xi_r)
        self.indices_c = np.array(sc["indices_c"])
    def __str__(self):
        s = f"Reference scheme\n"
        s += f"xi_c: {self.xi_c}\n"
        s += f"xi_r: {self.xi_r}\n"
        return s

class Integral(): 
    def __init__(self, f, a=0, b=pi/2):

        self.f = f
        self.reference_scheme = ReferenceScheme(2)
        self.interval_list = [Interval(a, b, self.reference_scheme)]
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
            # print(f"I_r, Ic={quad_int.I_r, quad_int.I_c}")
            # print(f"scipy={integrate.quad(self.f, quad_int.a, quad_int.b)[0]}")
            # print(f"trapz={(quad_int.b**2-quad_int.a**2)}")
            self.I_c += quad_int.I_c
            self.I_r += quad_int.I_r
            self.error_list.append(quad_int.error)
        self.error = np.sum(self.error_list)
        # self.error_list /= self.error

    def refine(self, it):
        interval_list = []
        for i_interval, quad_int in enumerate(self.interval_list):
            interval_list.extend(quad_int.split())
        self.interval_list = interval_list
            
        
        
        
        # new_interval_list = []
        # new_compute_list = []
        # print(self.error_list)
        # exit()
        # index_max = np.argmax(self.error_list)
        # refined_reference_scheme = ReferenceScheme(3)
            
        # if self.error_list[index_max]>0.5:
        #     if index_max == 0:
        #         # The first quadrature interval 
        #         quad_int = self.interval_list[0]
        #         for _int in quad_int.refine():
        #             new_interval_list.append(Interval(_int[0], _int[1], refined_reference_scheme))
        #             new_compute_list.append(True)
        #         new_interval_list += self.interval_list[1:]
        #         new_compute_list += self.compute_list[1:]
        #     elif index_max == len(self.error_list)-1:
        #         new_interval_list += self.interval_list[:-1]
        #         new_compute_list+= self.compute_list[:-1]
        #         quad_int = self.interval_list[-1]
        #         for _int in quad_int.refine():
        #             new_interval_list.append(Interval(_int[0], _int[1], refined_reference_scheme))
        #             new_compute_list.append(True)
        #     else:
        #         new_interval_list += self.interval_list[:index_max]
        #         new_compute_list +=self.compute_list[:index_max]
        #         quad_int = self.interval_list[index_max]
        #         for _int in quad_int.refine():
        #             new_interval_list.append(Interval(_int[0], _int[1], refined_reference_scheme))
        #             new_compute_list.append(True)
        #         new_interval_list +=self.interval_list[index_max+1:]
        #         new_compute_list +=self.compute_list[index_max+1:]
        # else:
        #     for quad_int in self.interval_list:
        #         list_intervals = quad_int.refine()
        #         for _int in list_intervals:
        #             new_interval_list.append(Interval(_int[0], _int[1], refined_reference_scheme))
        #             new_compute_list.append(True)
        # self.interval_list = new_interval_list
        # self.compute_list = new_compute_list

class Interval():
    def __init__(self, a, b, reference_scheme=ReferenceScheme(), previous_values=None, **kwargs):
        # print(f"Interval [{a}, {b}]")
        self.a = a
        self.b = b
        xi_a, xi_b = -1, 1
        Delta =(self.b-self.a)
        self.x_A, self.x_B = self.a, self.b
        
        self.reference_scheme = reference_scheme
        self.previous_values = previous_values

        self.n_c, self.n_r = reference_scheme.n_c, reference_scheme.n_r
        self.indices_c = reference_scheme.indices_c
        
        self.xi_power_c = reference_scheme.xi_powers_c
        self.xi_power_r = reference_scheme.xi_powers_r

        self.mat_c = np.linalg.inv(np.vander(reference_scheme.xi_c))
        self.mat_r = np.linalg.inv(np.vander(reference_scheme.xi_r))

        # self.w_c =  (self.b-self.a)*reference_scheme.w_c/2
        # self.w_r =  (self.b-self.a)*reference_scheme.w_r/2
        self.xi_c = reference_scheme.xi_c
        self.xi_r = reference_scheme.xi_r
        

        
        if self.previous_values is not None:
            # print(self.previous_values["type"])
            if self.previous_values["type"] == "left": # refinement of the first interval
                # print("1")
                # test to find the right substitution 
                # Substitution with x_p[0] as xi_c[0] and x_p[1] as xi_r[-1]
                # print(f"premier intervalle ({self.a}, {self.b})")
                alpha = (self.xi_r[-1]-self.xi_c[0])/(self.previous_values["x"][1]-self.previous_values["x"][0]) 
                beta = self.xi_r[-1]-alpha*self.previous_values["x"][1]
                xi_a, xi_b = alpha*self.a+beta, self.xi_r[-1]
            elif self.previous_values["type"] == "right": # refinement of the last interval
                # print("2")
                # print(f"dernier intervalle ({self.a}, {self.b})")
                alpha = (self.xi_c[-1]-self.xi_r[0])/(self.previous_values["x"][1]-self.previous_values["x"][0]) 
                beta = self.xi_c[-1]-alpha*self.previous_values["x"][1]
                xi_a, xi_b = self.xi_r[0], alpha*self.b+beta                # print([(xi-beta)/alpha for xi in [xi_a, xi_b]])
            elif self.previous_values["type"] == "middle":
                # print("3")
                # print("px", self.previous_values["x"])

                alpha = (self.xi_r[-1]-self.xi_r[0])/(self.previous_values["x"][2]-self.previous_values["x"][0])
                beta = self.xi_r[-1]-alpha*self.previous_values["x"][2]
                xi_a, xi_b = self.xi_r[0], self.xi_r[-1]
            else:
                jklljjlk
            self.x_a, self.x_b = (self.xi_r[0]-beta)/alpha, (self.xi_r[-1]-beta)/alpha
            self.x_A, self.x_B = (-1-beta)/alpha, (1-beta)/alpha
            Delta = 2/alpha
            self.x_r = (reference_scheme.xi_r-beta)/alpha
            self.x_c = (reference_scheme.xi_c-beta)/alpha
            # print(f"xi_a={xi_a}, xi_b={xi_b}")
        self.xia_power_c = np.array([(xi_a)**i for i in range(0, self.n_c+1)])[::-1]
        self.xia_power_r = np.array([(xi_a)**i for i in range(0, self.n_r+1)])[::-1]
        self.xib_power_c = np.array([(xi_b)**i for i in range(0, self.n_c+1)])[::-1]
        self.xib_power_r = np.array([(xi_b)**i for i in range(0, self.n_r+1)])[::-1]
        
        self.Delta = Delta
        # print(f"{Delta}/{self.x_B-self.x_A}")


        self.x_r = self.x_A+(self.x_B-self.x_A)*(reference_scheme.xi_r+1)/2
        self.x_c = self.x_A+(self.x_B-self.x_A)*(reference_scheme.xi_c+1)/2

        # print(self.x_c)

        self.Mat_c = (Delta/2)*reference_scheme.Mat_c
        self.Mat_r = (Delta/2)*reference_scheme.Mat_r

        # else:
        #     raise ValueError("No reference_scheme or nodes")

    def __str__(self):
        s = f"interval [{self.a} ; {self.b}]\n"
        if self.previous_values is not None:
            s += f"previous values: {self.previous_values}\n"
        return s

    def update(self, func, it):
        self.f_r = np.array([func(x) for x in self.x_r])
        f_c = self.f_r[self.indices_c]

        # self.I_c = np.sum(self.w_c*f_c)
        # self.I_r = np.sum(self.w_r*self.f_r)
        # # Pythonneries 
        # poly_c = lagrange(self.x_c, f_c)
        # poly_r = lagrange(self.x_r, self.f_r)
        # self.poly_c = lagrange(self.x_c, f_c)
        # self.poly_r = lagrange(self.x_r, self.f_r)      
        # poly_diff = (poly_r-poly_c)
        # poly_error = np.polyint(poly_diff)
        
        # Coefficients of the interpolating polynomial
        c_c = self.mat_c@f_c
        c_r = self.mat_r@self.f_r
        # Coefficients of the antideratives of the interpolating polynomial
        coeff_c = self.Mat_c@f_c
        coeff_r = self.Mat_r@self.f_r
        # Computation of the integrals

        self.I_c = coeff_c@(self.xib_power_c-self.xia_power_c)
        self.I_r = coeff_r@(self.xib_power_r-self.xia_power_r)
        self.error = np.abs((self.I_c-self.I_r)/self.I_r)
        
        # Computation of the errors 
        # x_c = np.hstack([self.a, self.x_c, self.b])
        # self.error_list_0 = np.abs(np.array([poly_error(x_c[i+1])-poly_error(x_c[i]) for i in range(self.n_c+1)]))
        # self.error_list_0 /= np.sum(self.error_list_0)
        self.error_list = np.abs(np.array([ (coeff_c@self.xi_power_c[i+1]-coeff_r@self.xi_power_r[i+1])-(coeff_c@self.xi_power_c[i]-coeff_r@self.xi_power_r[i])for i in range(self.n_c+1)]))
        # self.error_list /= np.sum(self.error_list)

        self.x_plot =np.linspace(self.a, self.b, 500)
        
        self.xi_plot = 2*(self.x_plot-self.x_A)/(self.Delta)-1
        
        self.v_plot = [func(xx) for xx in self.x_plot]
        plt.figure(it)
        # plt.plot(self.x_plot, poly_c(self.x_plot), "b", label="poly_c")
        
        plt.plot(self.x_r, self.f_r, "b.")
        plt.plot(self.x_c, [c_c@xi[1:] for xi in self.xi_power_c[1:-1]], "+")
        plt.plot(self.x_c, [c_r@xi[1:] for xi in self.xi_power_r[1:-1]], "+")
        # plt.plot(self.x_plot, poly_r(self.x_plot), "r", label="poly_f")
        # plt.plot(self.x_r, self.f_r, "r.")
        plt.plot(self.x_plot, self.v_plot, "m", label="Tau")
        plt.plot([self.a, self.a], [0, 3], color="k")
        plt.plot([self.b, self.b], [0, 3], color="k")
        def poly_r(xi):
            return np.sum([c_r[::-1][i]*xi**i for i in range(self.n_r)])
        p_plot = [poly_r(xi) for xi in self.xi_plot]
        plt.plot(self.x_plot, p_plot, "g", label="Tau")

        
        
        
        # plt.plot(self.x_plot, poly_diff(self.x_plot), "g", label="poly_error")
        # plt.plot(self.x_c, poly_diff(self.x_c), "g.")
        # plt.legend()
        # plt.figure(10+it)
        # plt.bar(range(self.n_c+1), self.error_list)
        # plt.plot(range(self.n_c+1), self.error_list,'r.-')
        # plt.show()

    def split(self):
        # detemine the number of nodes to add in each interval

        nb_sub_intervals = len(self.error_list)
        list_subintervals = [None]*nb_sub_intervals
        index_max = np.argmax(self.error_list)

        # First interval
        previous_values = {"x" : self.x_r[:2], "f": self.f_r[:2], "xi" : self.xi_r[:2], "type" : "left"}
        list_subintervals[0] = Interval(self.a, self.x_c[0], self.reference_scheme, previous_values)
        # Intermediate intervals
        for i in range(nb_sub_intervals-2):
            previous_values = {"x" : self.x_r[1+2*i:2+2*i+2], "f": self.f_r[1+2*i:2+2*i+2], "xi" : self.xi_r[1+2*i:2+2*i+2], "type" : "middle"}
            list_subintervals[i+1] = Interval(self.x_c[i], self.x_c[i+1], self.reference_scheme, previous_values)
        # Last interval
        previous_values = {"x" :  self.x_r[-2:], "f": self.f_r[-2:], "xi": self.xi_r[-2:], "type" : "right"}
        list_subintervals[nb_sub_intervals-1] = Interval(self.x_c[-1], self.b, self.reference_scheme, previous_values)
        return list_subintervals






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
