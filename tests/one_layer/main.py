import sys
sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.layers import one_layer

from pyPLANES.core.periodic_fem_problem import FemProblem, PeriodicFemProblem
from pyPLANES.core.periodic_pw_problem import PeriodicPwProblem
from pyPLANES.core.dgm_problem import DgmProblem


plot_solution = [True, True, True, False, False, True]

# Parameters of the simulation
frequencies = np.linspace(400, 5000., 1)
theta_d = 30
L = 5e-2
d = 5e-2
lcar = 1e-2

Wwood = "Wwood"
melamine = "melamine"
material = "Wwood"

# material = "melamine"
# material = "melamine_eqf"
# material = "Air"

name_project = "one_layer"
ml = [(material, d)]
termination = "transmission"
# termination = "rigid"
# plot_solution = [False]*6
# plot_solution = [True]*6

method = "FEM"

bc_bottom = "bottom"
bc_right = "Periodicity"
bc_left = "Periodicity"
bc_top = "top"

# ml = [("Wwood", d), ("melamine", d), ("Wwood", d)]
# ml = [("melamine", d),("Wwood", d)]
# ml = [("Wwood", 2*d)]
# ml = [("Wwood", d),("Air", d)]
# ml = [ (melamine, d), (wood, d)]
# ml = [(wood, d)]
# ml = [(material, 2*d)]
# ml = [("Air", d), ("melamine", d)]

material = "melamine"
material = "Air"
# material = "Wwood"

ml = [(material, d)]
# ml = [("Air", d)]
# ml = [("melamine", d)]
ml = [("Air", d), (melamine, d)]

global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="global", verbose=False, print_result=True)
global_method.resolution()


recursive_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="JAP", verbose=False, print_result=True)
recursive_method.resolution()

one_layer(name_mesh=material + "_FEM", L=L, d=d, lcar=lcar, mat=material, method="FEM",  BC=[bc_bottom, bc_right, bc_top, bc_left])

# ml = [("melamine", d), (melamine+"_FEM", d)]
ml = [(material + "_FEM" , d)]

eTMM_method = PeriodicPwProblem(ml=ml, name_project=name_project, theta_d=theta_d, order=6, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False, print_result=True)
eTMM_method.resolution()

if any(plot_solution):
    plt.show()
