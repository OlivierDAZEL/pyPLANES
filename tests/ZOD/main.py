import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.fem_problem import FemProblem
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.ZOD import ZOD


frequencies = (60., 1500., 1)
name_mesh = "ZOD"
L = 1
d = 1
lcar = 1
h = 0.2

ZOD(name_mesh, L, d, d, lcar, lcar/2, h, "Air", "Rigid Wall")

order = 2
plot_results = [True, True, True, False, False, True]
# plot_results = [False]*6


theta_d = 0.000000
name_project = "ZOD"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=2, frequencies=frequencies, plot_results=plot_results)
problem.resolution()


ml = [("Air", d+L+h)]
problem = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results,termination="rigid")
problem.resolution()




if any(plot_results):
    plt.show()
