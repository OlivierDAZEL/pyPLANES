import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.periodic_fem_problem import FemProblem, PeriodicFemProblem
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.layers import one_layer


# Parameters of the simulation 
frequencies = np.linspace(2000., 5000., 1)
theta_d = 40.000

L = 5e-2
d = 5e-2
lcar = 1e-2
material = "Wwood"
# material = "melamine"
material = "Air"

name_project = "one_layer"
ml = [(material, d)]
termination = "rigid" 

plot_solution = [True, True, True, False, False, False]
# plot_solution = [False]*6
# plot_solution = [True]*6

one_layer(name_project, L, d, lcar, material, "Rigid Wall")

global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="global", verbose=False)
global_method.resolution()

fem = PeriodicFemProblem(name_project=name_project, name_mesh=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=True)
fem.resolution()




if any(plot_solution):
    plt.show()

   