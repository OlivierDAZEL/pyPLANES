import sys
sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.gmsh.templates.layers import one_layer

from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.core.periodic_pw_problem import PeriodicPwProblem


plot_solution = [True, True, True, False, False, False]
# Parameters of the simulation
frequencies = np.linspace(100, 5000., 1)
theta_d = 30
L = 5e-2
d = 5e-2
lcar = 1e-2

termination = "transmission"
termination = "rigid"

ml = [("Air", d), ("melamine", d)]

global_method = PwProblem(ml=ml, name_project="one_layer", theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="global", verbose=False, print_result=True)
global_method.resolution()


recursive_method = PwProblem(ml=ml, name_project="one_layer", theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="JAP", verbose=False, print_result=True)
recursive_method.resolution()

material = "melamine"
mesh_EF = material + "_FEM"
one_layer(name_mesh=mesh_EF, L=L, d=d, lcar=lcar, mat=material)

# ml = [("melamine", d), (melamine+"_FEM", d)]
ml = [("Air", d), (material + "_FEM" , d)]

eTMM_method = PeriodicPwProblem(ml=ml, name_project="one_layer", theta_d=theta_d, order=3, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False, print_result=True)
eTMM_method.resolution()

if any(plot_solution):
    plt.show()
