import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.fem_problem import PeriodicFemProblem
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.layers import one_layer

frequencies = (100., 3500., 20)
name_mesh = "one_layer"
L = 0.01
d = 0.02
lcar = 0.01

termination = "rigid"
termination = "transmission"
material = "Air"
material = "pem_benchmark_1"
# material = "eqf_benchmark_1"
# material = "Wwood"
one_layer(name_mesh, L, d, lcar, material, "Rigid Wall")

# order = 3
plot_results = [True, True, True, False, False, False]
plot_results = [False]*6

# incident_ml = [("rubber", 0.0002)] ; shift_pw = -incident_ml[0][1]

# theta_d = 0.00000
name_project = "one_layer"
# problem = PeriodicFemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=2, frequencies=frequencies, plot_results=plot_results)
# result_pyPLANES = problem.resolution()

theta_d = 30.000000001
ml = [(material, d)]
global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results,termination=termination, method="global", verbose=False)
global_method.resolution()

recursive_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results,termination=termination, method="recursive", verbose=False)
recursive_method.resolution()

if any(plot_results):
    plt.show()

from pymls import from_yaml, Solver, Layer, backing
from mediapack import Air

mat_pymls = from_yaml(material+".yaml")

solver_pymls = Solver()
solver_pymls.layers = [
    Layer(mat_pymls, d),
]

solver_pymls.backing = backing.transmission
# solver_pymls.backing = backing.rigid
R_pymls, f_pymls = [], []
T_pymls = []

for _f in global_method.frequencies:
    f_pymls.append(_f)
    R_pymls.append(solver_pymls.solve(_f, theta_d)["R"][0])
    T_pymls.append(solver_pymls.solve(_f, theta_d)["T"][0])

plt.figure()
global_method.plot_results()
recursive_method.plot_results()
plt.figure(1)
plt.plot(f_pymls, [_.real for _ in R_pymls], 'r',label="pymls")
plt.plot(f_pymls, [_.imag for _ in R_pymls], 'b',label="pymls")
plt.legend()
plt.figure(2)
plt.plot(f_pymls, [_.real for _ in T_pymls], 'r',label="pymls")
plt.plot(f_pymls, [_.imag for _ in T_pymls], 'b',label="pymls")
plt.legend()
plt.show()

# print("result_pymls     = {}".format(result_pymls["R"][0]))
# FEM = np.loadtxt("one_layer.FEM.txt")
# PW = np.loadtxt("one_layer.PW.txt")
# plt.plot(PW[:,0],PW[:,1], "b", label="PW")
# plt.plot(FEM[:, 0],FEM[:, 1], "r.", label="FEM")
# plt.show()        