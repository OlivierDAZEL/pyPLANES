import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.classes.fem_problem import FemProblem
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.ZOD import ZOD


frequencies = (600., 1500., 1)
name_mesh = "ZOD"
L = 0.1
d = 0.1
lcar = 0.1

ZOD(name_mesh, L, d, d, lcar, "Air", "Rigid Wall")

order = 2
plot_results = [True, True, True, False, False, False]
# plot_results = [False]*6


theta_d = 10.000000
name_project = "ZOD"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=2, frequencies=frequencies, plot_results=plot_results)
result_pyPLANES = problem.resolution()

ml = [("Air", d)]
S_PW = Solver_PW(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results)
result_pyPLANESPW = S_PW.resolution(theta_d)

if any(plot_results):
    plt.show()
