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
