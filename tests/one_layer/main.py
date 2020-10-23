import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.fem_problem import FemProblem
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.layers import one_layer

frequencies = (600., 1500., 1)
name_mesh = "one_layer"
L = 0.01
d = 0.02
lcar = 0.01

material = "Air"
material = "pem_benchmark_1"
material = "Wwood"
one_layer(name_mesh, L, d, lcar, material, "Transmission")

# order = 3
plot_results = [False, True, True, False, False, False]
# plot_results = [False]*6

# incident_ml = [("rubber", 0.0002)] ; shift_pw = -incident_ml[0][1]

theta_d = 0.000001
name_project = "one_layer"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=2, frequencies=frequencies, plot_results=plot_results)
result_pyPLANES = problem.resolution()


ml = [(material, d)]
problem = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results,termination="transmission")
problem.resolution()

if any(plot_results):
    plt.show()

# from pymls import from_yaml, Solver, Layer, backing
# from mediapack import Air
# # pem_benchmark1 = from_yaml(material+".yaml")
# solver_pymls = Solver()
# solver_pymls.layers = [
#     Layer(Air, d),
# ]

# solver_pymls.backing = backing.transmission
# result_pymls =  solver_pymls.solve(frequencies[0], theta_d)

# print((result_pymls["R"][0]))


# print("result_pymls     = {}".format(result_pymls["R"][0]))
# FEM = np.loadtxt("one_layer.FEM.txt")
# PW = np.loadtxt("one_layer.PW.txt")
# plt.plot(PW[:,0],PW[:,1], "b", label="PW")
# plt.plot(FEM[:, 0],FEM[:, 1], "r.", label="FEM")
# plt.show()        