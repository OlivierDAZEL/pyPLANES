import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
from pyPLANES.core.periodic_fem_problem import FemProblem, PeriodicFemProblem
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.layers import one_layer

# Parameters of the simulation 
frequencies = np.linspace(1000., 5000., 1)
theta_d = 0.000

L = 5e-2
d = 5e-2
lcar = 5e-2
material = "Wwood"
material = "melamine"
# material = "melamine_eqf"
material = "Air"

name_project = "one_layer"
ml = [(material, d)]
termination = "transmission" 
termination = "rigid" 

plot_solution = [True, True, True, False, False, False]
# plot_solution = [False]*6
# plot_solution = [True]*6

# one_layer(name_project, L, d, lcar, material, "Transmission")
one_layer(name_project, L, d, lcar, material, "Rigid Wall")

global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="global", verbose=False)
global_method.resolution()

fem = PeriodicFemProblem(name_project=name_project, name_mesh=name_project, order=3, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False)
fem.resolution()

# print("abs                = {}".format(1-np.abs(global_method.R)**2-np.abs(global_method.T)**2))
print("abs                = {}".format(1-np.abs(global_method.R)**2))


if any(plot_solution):
    plt.show()

   
=======
from pyPLANES.classes.fem_problem import FemProblem
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.layers import one_layer


frequencies = (600., 1500., 1)
name_mesh = "one_layer"
L = 0.01
d = 0.02
lcar = 0.01

one_layer(name_mesh, L, d, lcar, "pem_benchmark_1", "Rigid Wall")

order = 3
plot_results = [True, True, True, False, False, False]
# plot_results = [False]*6

incident_ml = [("rubber", 0.0002)] ; shift_pw = -incident_ml[0][1]

theta_d = 60.000000
name_project = "one_layer"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=2, frequencies=frequencies, plot_results=plot_results, incident_ml=incident_ml)
result_pyPLANES = problem.resolution()

ml = [("rubber", 0.0002), ("pem_benchmark_1", d)]
S_PW = Solver_PW(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_results=plot_results)
result_pyPLANESPW = S_PW.resolution(theta_d)

if any(plot_results):
    plt.show()


FEM = np.loadtxt("one_layer.FEM.txt")
PW = np.loadtxt("one_layer.PW.txt")
plt.plot(PW[:,0],PW[:,1], "b", label="PW")
plt.plot(FEM[:, 0],FEM[:, 1], "r.", label="FEM")
plt.show()
>>>>>>> bf5bfed38de027d5d45b997bd157b85e89937121
