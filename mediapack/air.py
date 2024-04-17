#! /usr/bin/env python
# -*- coding:utf8 -*-
#
# air.py
#
# This file is part of pymls, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2017
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

from .medium import Medium


class Air(Medium):

    name = 'Air'
    T0 = 20.0
    P0 = 101325.0
    H = 45.0
    MEDIUM_TYPE = 'fluid'
    MODEL = MEDIUM_TYPE
    T = T0
    H = H
    P = P0
    TK = T0+273.15
    R = 287.031 # Universal gas constant (J.K^-1.kg^-1)
    f = 1.00062+3.14E-8*P0+5.6E-7*T0**2
    Psv = np.exp((1.2378847E-5)*TK**2-(1.9121316E-2)*TK+33.93711047-(6.3431645E3)/TK)
    xc=0.0004
    xw=(H/100)*Psv/P0*f;
    Z = 1-(P/TK)*(1.58123E-6-2.9331E-8*T0+1.1043E-10*T0**2+(5.707E-6-2.051E-8*T0)*xw+(1.9898E-4-2.376E-6*T0)*xw**2)+((P/TK)**2)*(1.83E-11-0.765E-8*xw**2)
    
    rho =(3.48349+1.44*(xc-0.0004))*1E-3*(P0/(Z*TK))*(1-0.378*xw);
    
    c= 331.5024+0.603055*T0-0.000528*T0**2+(51.471935+0.1495874*T0-0.000782*T0**2)*xw+(-1.82E-7+3.73E-8*T0-2.93E-10*0*0)*P0+(-85.20931-0.228525*T0+5.91E-5*T0**2)*xc-2.835149*xw**2-2.15E-13*P0**2+29.179762*xc**2+0.000486*xw*P0*xc
    
    gamma = 1.400822-1.75E-5*T0-1.73E-7*T**2+(-0.0873629-0.0001665*T0-3.26E-6*T0**2)*xw+(2.047E-8-1.26E-10*T0+5.939E-14*T0**2)*P0+(-0.1199717-0.0008693*T0+1.979E-6*T0**2)*xc-0.01104*xw**2-3.478E-16*P0**2+0.0450616*xc**2+1.82E-6*xw*P0*xc
    
    eta_fluid = (84.986+7*TK+(113.157-1*TK)*xw-3.7501E-3*TK*TK-100.015*xw**2)*1E-8
    mu = eta_fluid
    lam=4186.8*((60.054+1.846*TK+2.06E-6*TK**2+(40-1.775E-4*TK)*xw)*1E-8)
    
    Cp = 4186.8*(0.251625-9.2525E-5*TK+2.1334E-7*TK*TK-1.0043E-10*TK**3+(0.12477-2.283E-5*TK+1.267E-7*TK**2)*xw+(0.01116+4.61E-6*TK+1.74E-8*TK**2)*xw**2)
    
    Pr=eta_fluid*Cp/lam
    Cv = Cp - R
    Z = rho*c
    K_fluid=gamma*P
    K_fluid_norm=P
    K = gamma*P
    dm=3.8e-10 # molecule size (m)
    m_mol=28e-3 # molar mass (kg/mol)

    # T = 293.15  # reference temperature [K]
    # P = 1.01325e5  # atmospheric Pressure [Pa]
    # gamma = 1.400  # polytropic coefficient []
    # lambda_ = 0.0262  # thermal conductivity [W.m^-1.K^-1]
    # mu = 0.1839e-4  # dynamic viscosity [kg.m^-1.s^-1]
    # Pr = 0.710  # Prandtl's number []
    # molar_mass = 0.29e-1  # molar mass [kg.mol^-1]
    # rho = 1.213  # density [kg.m^-3]
    # C_p = 1006  # (mass) specific heat capacity as constant pressure [J.K^-1]

    # K = gamma*P  # adiabatic bulk modulus
    # c = np.sqrt(K/rho)  # adiabatic sound speed
    # Z = rho*c  # characteristic impedance
    # C_v = C_p/gamma  # (mass) specific heat capacity as constant volume [J.K^-1]
    nu = mu/rho  # kinematic viscosity [m.s^-2]
    nu_prime = nu/Pr  # viscothermal losses
    
    
    def __str__(self):
        txt = "Air properties\n"
        txt = "model: Full (Matelys)"
        return txt

    def from_yaml(self, *args, **kwargs):
        pass

    def update_frequency(self, *args, **kwargs):
        pass
