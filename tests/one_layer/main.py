import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt
    
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

   
