import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
from pyPLANES.core.fem_problem import FemProblem
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.ZOD import ZOD, ZOD_curved


frequencies = (60., 1500., 1)
name_mesh = "ZOD"
L = 1
d = 1
lcar = 1
h = 0.1
h_c = -0.0

ZOD(name_mesh, L, d, d, lcar/5, lcar/5, h, "Air", "Rigid Wall")

order = 2
plot_results = [True, True, True, False, False, False]
plot_results = [False]*6


interface_ml = {"ml":[("Air", h)]}
theta_d = 0.000000
name_project = "ZOD"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=2, frequencies=frequencies, plot_results=plot_results, interface_ml=interface_ml)
problem.resolution()


ml = [("Air", d+L+h)]
problem = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results,termination="rigid")
problem.resolution()

if any(plot_results):
    plt.show()

# ZOD_curved(name_mesh, L, d, d, lcar, lcar/3, h, h_c, "Air", "Rigid Wall")

=======
from pyPLANES.classes.fem_problem import FemProblem
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.ZOD import ZOD


frequencies = (600., 1500., 1)
name_mesh = "ZOD"
L = 0.1
d = 0.1
lcar = 0.05

ZOD(name_mesh, L, d, d, lcar, 0.02, "Air", "Rigid Wall")

order = 2
plot_results = [True, True, True, False, False, False]
# plot_results = [False]*6


theta_d = 10.000000
name_project = "ZOD"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=2, frequencies=frequencies, plot_results=plot_results)

# for _ent in problem.model_entities:
#     print(_ent)



# result_pyPLANES = problem.resolution()

# ml = [("Air", d)]
# S_PW = Solver_PW(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results)
# result_pyPLANESPW = S_PW.resolution(theta_d)

# if any(plot_results):
#     plt.show()
>>>>>>> bf5bfed38de027d5d45b997bd157b85e89937121
