#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# PW_methods_examples.py
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
# This scripts compares the 3 PW methods 

import sys
sys.path.insert(0, "../..")


from mediapack import Air
import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.core.result import Results, Result, Test


name_project="absorption_melanine_frequency"
plot_solution = [True, True, True, False, False, False]
verbose = [True, False][1]
# Parameters of the simulation
theta_d = 60.00000

d = 5.e-2
frequency = np.linspace(10,5e3,200)

termination = "rigid"
material = "melamine"


ml = [(material, d)]

global_method = PwProblem(ml=ml, name_project=name_project+"_GM", theta_d=theta_d, frequencies=frequency, termination=termination, method="global", verbose=verbose, print_result=True)
global_method.resolution()


recursive_method = PwProblem(ml=ml, name_project=name_project+"_JAP", theta_d=theta_d, frequencies=frequency, termination=termination, method="JAP", verbose=verbose, print_result=True)
recursive_method.resolution()

characteristic_method = PwProblem(ml=ml, name_project=name_project+"_carac", theta_d=theta_d, frequencies=frequency,termination=termination, method="characteristics", verbose=verbose, print_result=True)
characteristic_method.resolution()


GM = Result(_in="out/" + name_project + "_GM")
JAP = Result(_in="out/" + name_project + "_JAP")
carac = Result(_in="out/" + name_project + "_carac")

plt.figure()
plt.plot(GM.f,GM.abs,"b")
plt.plot(JAP.f,JAP.abs,"b.")
plt.plot(carac.f,carac.abs,"b+")


plt.show()