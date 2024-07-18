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
# from .quadrature_schemes import import_scheme
from matplotlib.patches import Rectangle 
from .reference_scheme import ReferenceScheme

class Interval():
    def __init__(self, a, b, reference_scheme=None, f_a=None, f_b=None):
        # print(f"Interval [{a}, {b}]")
        self.a = self.A = a
        self.b = self.B = b
        self.f_a, self.f_b = f_a, f_b
        self.reference_scheme = reference_scheme
        if reference_scheme is None:
            raise NameError("Reference scheme is missing")
        self.typ = reference_scheme.typ
        
        self.x_c = self.A+(self.B-self.A)*(reference_scheme.xi_c+1)/2.
        self.x_r = self.A+(self.B-self.A)*(reference_scheme.xi_r+1)/2.

        self.matrix_Poly_c = ((self.B-self.A)/2)*reference_scheme.matrix_Poly_c
        self.matrix_Poly_r = ((self.B-self.A)/2)*reference_scheme.matrix_Poly_r

    def __str__(self):
        s = f"[{self.a} ; {self.b}]"
        return s


    def update_f_values(self, func):
        self.f_r = np.array([func(x) for x in self.x_r])
        self.f_c = self.f_r[::2]
        
    def update(self, func):
        self.update_f_values(func)
        # self.coeff_c = (self.matrix_poly_c@self.f_c)
        # self.coeff_r = (self.matrix_poly_r@self.f_r)
        
        # zeros are added to the coefficients of the coarse scheme
        self.coeff_c = np.hstack([self.matrix_Poly_c@self.f_c, [0]*(self.reference_scheme.n_r-self.reference_scheme.n_c)])
        self.coeff_r = (self.matrix_Poly_r@self.f_r)
        
        # Computation of the integrals

        self.I_c = self.coeff_c@(self.reference_scheme.xib_powers-self.reference_scheme.xia_powers)
        self.I_r = self.coeff_r@(self.reference_scheme.xib_powers-self.reference_scheme.xia_powers)
        # self.error = np.abs((self.I_c-self.I_r)/self.I_r)
        
        # Computation of the errors
        if self.typ == "GK": 
            x_c = np.hstack([self.a, self.x_c, self.b])
        elif self.typ in ["CC", "PV"]:
            x_c = self.x_c
        else:
            raise NameError("Unknown quadrature scheme")

        if self.typ in ["CC", "PV"]:
            self.error_list = np.array([ (self.coeff_c@self.reference_scheme.xi_powers[i+1]-self.coeff_r@self.reference_scheme.xi_powers[i+1])-(self.coeff_c@self.reference_scheme.xi_powers[i]-self.coeff_r@self.reference_scheme.xi_powers[i])for i in range(self.reference_scheme.n_c-1)])
            self.error = np.sum(self.error_list)
        elif self.typ == "GK":
            raise NotImplementedError()
        else:
            raise NameError("Unknown quadrature scheme")

    def split(self,refined_schemes):
        index_max = np.argmax(np.abs(self.error_list))
        print(f"Splitting interval {index_max} on {len(self.error_list)-1}")
        refined_scheme = refined_schemes["3"]
        
        if index_max == 0:
            list_subintervals = []
            list_subintervals.append(Interval(self.x_r[0], self.x_r[1], refined_scheme,self.f_r[0], self.f_r[1]))
            list_subintervals.append(Interval(self.x_r[1], self.x_r[2], refined_scheme,self.f_r[1], self.f_r[2]))            
            x, f = self.x_r[2:], self.f_r[2:]
            if len(x) > 1:
                reference_scheme = ReferenceScheme("PV", x=x, f=f)
                list_subintervals.append(Interval(x[0], x[-1], reference_scheme))

        elif index_max == len(self.error_list)-1:
            list_subintervals = []
            x, f = self.x_r[:-2], self.f_r[:-2]
            if len(x) > 1:
                reference_scheme = ReferenceScheme("PV", x=x, f=f)
                list_subintervals.append(Interval(x[0], x[-1], reference_scheme))
            list_subintervals.append(Interval(self.x_r[-3], self.x_r[-2], refined_scheme,self.f_r[-3], self.f_r[-2]))
            list_subintervals.append(Interval(self.x_r[-2], self.x_r[-1], refined_scheme,self.f_r[-2], self.f_r[-1]))    
        else:
            list_subintervals = []
            indices = slice(1+2*index_max)
            x, f = self.x_r[indices], self.f_r[indices]
            reference_scheme = ReferenceScheme("PV", x=x, f=f)
            list_subintervals.append(Interval(x[0], x[-1], reference_scheme))
            index_a, index_b =  2*index_max , 2*index_max+1
            list_subintervals.append(Interval(self.x_r[index_a], self.x_r[index_b], refined_scheme,self.f_r[index_a], self.f_r[index_b]))
            index_a, index_b = 2*index_max+1 , 2*index_max+2
            list_subintervals.append(Interval(self.x_r[index_a], self.x_r[index_b], refined_scheme,self.f_r[index_a], self.f_r[index_b]))
            indices = slice(2*index_max+2, self.reference_scheme.n_r)
            x, f = self.x_r[indices], self.f_r[indices]
            reference_scheme = ReferenceScheme("PV", x=x, f=f)
            list_subintervals.append(Interval(x[0], x[-1], reference_scheme))

        return list_subintervals
            

            
            
            










        # if self.typ == "CC":
        #     nb_sub_intervals = len(self.x_c)-1
        #     list_subintervals = []
        #     for i in range(nb_sub_intervals):
        #         if i == index_max:
        #             previous_values = {"x_r" : self.x_r[:2], "f": self.f_r[:2], "xi" : self.xi_r[:2], "type" : "left"}
        #         else:
        #             indices = slice(1+2*i,1+2*i+3) 
        #             x, f = self.x_r[indices], self.f_r[indices]
        #             previous_values = ReferenceScheme("PV", x=x, f=f)
        #             exit()    
                
                
        #         list_subintervals.append(Interval(self.x_c[i], self.x_c[i+1], self.typ, self.reference_scheme, previous_values))
            
        #     for inter in list_subintervals:
        #         print(inter)
        #     exit()
        # elif self.typ == "GK":
        #     # detemine the number of nodes to add in each interval
        #     nb_sub_intervals = len(self.x_c)+1
        #     list_subintervals = [None]*nb_sub_intervals
            
            
        #     # First interval
        #     previous_values = {"x_r" : self.x_r[:2], "f": self.f_r[:2], "xi" : self.xi_r[:2], "type" : "left"}
        #     list_subintervals[0] = Interval(self.a, self.x_c[0], self.typ, self.reference_scheme, previous_values)
        #     # Intermediate intervals
        #     for i in range(nb_sub_intervals-2):
        #         previous_values = {"x_r" : self.x_r[1+2*i:1+2*i+3], "f": self.f_r[1+2*i:1+2*i+2], "xi" : self.xi_r[1+2*i:1+2*i+3], "type" : "middle"}
        #         list_subintervals[i+1] = Interval(self.x_c[i], self.x_c[i+1], self.typ, self.reference_scheme, previous_values)
        #     # Last interval
        #     previous_values = {"x_r" :  self.x_r[-2:], "f": self.f_r[-2:], "xi": self.xi_r[-2:], "type" : "right"}
        #     list_subintervals[nb_sub_intervals-1] = Interval(self.x_c[-1], self.b, self.typ, self.reference_scheme, previous_values)
        # else:
        #     raise NameError("Unknown quadrature scheme")

    def plot_polynomials(self, fig=None):
        x_plot =np.linspace(self.A, self.B, 500)
        xi_plot =np.linspace(-1, 1, 500)
        c_c = (self.reference_scheme.matrix_poly_c@self.f_c)
        c_r = (self.reference_scheme.matrix_poly_r@self.f_r)
        c_plot = (np.array([np.sum([c_c[i]*xi**i for i in range(self.reference_scheme.n_c)]) for xi in xi_plot]))
        r_plot = (np.array([np.sum([c_r[i]*xi**i for i in range(self.reference_scheme.n_r)]) for xi in xi_plot]))
        plt.plot(x_plot,c_plot,"g")
        plt.plot(x_plot,r_plot,"m")

    def plot_error_function(self, fig):
        x_plot =np.linspace(self.A, self.B, 500)
        xi_plot =np.linspace(-1, 1, len(x_plot))
        v_plot = [np.exp(xx) for xx in x_plot]
        # plt.plot(x_plot, v_plot, "b")
        c_c = (self.matrix_poly_c@self.f_c)
        c_r = (self.matrix_poly_r@self.f_r)
        erreur_c_plot = (np.array([np.sum([c_c[i]*xi**i for i in range(self.n_c)]) for xi in xi_plot])-v_plot)
        erreur_r_plot = (np.array([np.sum([c_r[i]*xi**i for i in range(self.n_r)]) for xi in xi_plot])-v_plot)
        plt.plot(x_plot,erreur_c_plot,"g")
        plt.plot(x_plot,erreur_r_plot,"m")

    def plot_error_on_intervals(self, ax):
        if self.typ in ["CC","PV"]: 
            error_list = self.error_list
            for i, error in enumerate(error_list):
                if error > 0:
                    facecolor = "blue"
                else:
                    facecolor = "red"
                ax.add_patch(Rectangle((self.x_c[i], 0), self.x_c[i+1]-self.x_c[i], abs(error), edgecolor = 'black', facecolor = facecolor, fill=True,lw=1))
        else:
            raise NameError("Unknown quadrature scheme")


    def plot_grid(self,i):
        plt.plot(self.x_r,i*np.ones(self.reference_scheme.n_r),"b.")
        plt.plot(self.x_r[0],i,"r.")
        plt.plot(self.x_r[-1],i,"r.")



