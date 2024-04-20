#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# tube_air_FEM.py
#
# This file is part of pyplanes, a software distributed under the MIT license.
# For any question, please contact one of the authors cited below.
#
# Copyright (c) 2024
# 	Olivier Dazel <olivier.dazel@univ-lemans.fr>
# 	Mathieu Gaborit <gaborit@kth.se>
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
# This scripts compares the PW and hybrid PW/FEM methods for a simple tube problem





import sys
sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.gmsh.templates.layers import one_layer
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.core.periodic_pw_problem import PeriodicPwProblem
from pyPLANES.core.fem_problem import FemProblem
from pyPLANES.core.result import Results
from mediapack import Air


plot_solution = [True, True, True, False, False, False]
frequencies = np.linspace(30, 500., 1)

L = 1
d = 1
lcar = 0.1
condensation = True
order_geometry = 1
termination = "rigid"


ml = [("Air", d)]

theta_d = 0.00000
name_project = "tube_air"

ml = [("Air", d)]
global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies,termination=termination, method="global", verbose=False, print_result=True,plot_solution=plot_solution)
global_method.resolution()

one_layer(name_mesh="Air_FEM", L=L, d=d, lcar=lcar, mat="Air", BC = ["bottom", "Periodicity", "top", "Periodicity"], order_geometry = order_geometry)

ml = [("Air_FEM", None)]

Periodic_method = PeriodicPwProblem(ml=[("Air_FEM",None)], name_project=name_project, theta_d=theta_d, frequencies=frequencies,plot_solution=plot_solution,termination=termination)
Periodic_method.resolution()

if any(plot_solution):
    plt.xlabel("Position")
    plt.ylabel("Presure")
    plt.show()
