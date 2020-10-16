import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.fem_problem import FemProblem
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.layers import one_layer

frequencies = (6000., 1500., 1)
name_mesh = "one_layer"
L = 0.01
d = 0.02
lcar = 0.01

one_layer(name_mesh, L, d, lcar, "Air", "Rigid Wall")

# order = 3
plot_results = [True, True, True, False, False, False]
# # plot_results = [False]*6

# incident_ml = [("rubber", 0.0002)] ; shift_pw = -incident_ml[0][1]

theta_d = 60.000000
name_project = "one_layer"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=2, frequencies=frequencies, plot_results=plot_results)
result_pyPLANES = problem.resolution()

ml = [("Air", d)]
problem = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results)
problem.resolution()



if any(plot_results):
    plt.show()


# FEM = np.loadtxt("one_layer.FEM.txt")
# PW = np.loadtxt("one_layer.PW.txt")
# plt.plot(PW[:,0],PW[:,1], "b", label="PW")
# plt.plot(FEM[:, 0],FEM[:, 1], "r.", label="FEM")
# plt.show()        